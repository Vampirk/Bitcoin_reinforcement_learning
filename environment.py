import gym
from gym import spaces
import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple
import logging
from sklearn.preprocessing import MinMaxScaler
from config import DATA_PATH, INITIAL_ACCOUNT_BALANCE, MAKER_FEE_PCT, TAKER_FEE_PCT, LEVERAGE, NEURAL_NET_CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# 데이터 로드
DATA = pd.read_csv(DATA_PATH)

# Min-Max 스케일링 객체 생성
scaler = MinMaxScaler()


class Action(Enum):
    BUY = 0
    SELL = 1
    LIQUIDATE = 2
    HOLD = 3

class FuturesTradingEnv(gym.Env):
    def __init__(self):
        super(FuturesTradingEnv, self).__init__()

        # 데이터 로드
        self.data = pd.DataFrame(scaler.fit_transform(DATA), columns=DATA.columns)
        self.initial_account_balance = INITIAL_ACCOUNT_BALANCE
        self.current_step = 0
        self.maker_fee_pct = MAKER_FEE_PCT
        self.taker_fee_pct = TAKER_FEE_PCT
        self.leverage = LEVERAGE
        self.position = 0
        self.entry_price = 0
        self.account_balance = INITIAL_ACCOUNT_BALANCE

        # 강제 청산을 위한 손실 기준치 설정 (예: -10%)
        self.force_sell_threshold = -1

        # 상태 및 행동 공간 정의
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(NEURAL_NET_CONFIG['input_size'],), dtype=np.float32)
        self.action_space = spaces.Discrete(len(Action))

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # 행동 수행
        self._take_action(action)
        reward = self._get_reward()

        # 다음 상태 및 보상 계산
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            done = False
        obs = self._next_observation()

        return obs, reward, done, {}

    def _next_observation(self) -> np.ndarray:
        """현재 상태(가격, 기술 지표 등)를 반환합니다."""
        return self.data.iloc[self.current_step].values

    def _observe_wait(self, current_price: float) -> None:
        """관망 시 손익률을 계산하고 강제 청산 여부를 확인합니다."""
        # 현재 손익률 계산
        if self.position > 0:
            percentage = (current_price - self.entry_price) / self.entry_price
        elif self.position < 0:
            percentage = (self.entry_price - current_price) / current_price
        else:
            percentage = 0

        logger.info("관망으로 행동합니다.")
        logger.info(f"지금 손해 또는 이익: {percentage * self.leverage:.2%}")

        # 손실이 일정 기준치 이상인 경우 강제 청산
        if percentage <= self.force_sell_threshold:
            logger.info("손실이 일정 기준치 이상이므로 강제 청산합니다.")
            self._liquidate_position(current_price)

    def _take_action(self, action: int) -> None:
        """지정된 행동을 수행합니다."""
        current_price = self.data.iloc[self.current_step]['close']
        action = Action(action)

        try:
            if action == Action.BUY:
                self._buy(current_price)
            elif action == Action.SELL:
                self._sell(current_price)
            elif action == Action.LIQUIDATE:
                self._liquidate_position(current_price)
            elif action == Action.HOLD:
                self._observe_wait(current_price)
            else:
                raise ValueError(f"Invalid action: {action}")
        except Exception as e:
            logger.error(f"Error occurred during action {action}: {e}")

    def _buy(self, current_price: float) -> None:
        """매수 주문을 실행합니다."""
        if self.position == 0:
            purchase_quantity = self.account_balance / current_price
            purchase_quantity /= 2
            trade_amount = purchase_quantity * current_price

            # 거래 수수료 계산
            fee = trade_amount * self.maker_fee_pct

            # 새로운 포지션 업데이트
            self.position += purchase_quantity
            self.entry_price = current_price

            # 계좌 잔고 업데이트
            self.account_balance -= (trade_amount + fee)

            logger.info(f"매수 주문: 가격={current_price}, 수량={purchase_quantity}, 수수료={fee}, 잔고={self.account_balance:.2f}, 포지션={self.position}")
        else:
            logger.info("이미 매수 포지션을 보유 중입니다. 관망으로 행동 변경합니다.")
            self._observe_wait(current_price)

    def _sell(self, current_price: float) -> None:
        """매도 주문을 실행합니다."""
        if self.position == 0:
            purchase_quantity = self.account_balance / current_price
            purchase_quantity /= -2
            trade_amount = purchase_quantity * current_price

            # 거래 수수료 계산
            fee = trade_amount * self.maker_fee_pct

            # 새로운 포지션 업데이트
            self.position += purchase_quantity
            self.entry_price = current_price

            # 계좌 잔고 업데이트
            self.account_balance += (trade_amount + fee)

            logger.info(f"매도 주문: 가격={current_price}, 수량={purchase_quantity}, 수수료={fee}, 잔고={self.account_balance:.2f}, 포지션={self.position}")
        else:
            logger.info("이미 매수 포지션을 보유 중입니다. 관망으로 행동 변경합니다.")
            self._observe_wait(current_price)

    def _liquidate_position(self, current_price: float) -> None:
        """현재 포지션을 청산합니다."""
        if self.position != 0:
            sell_quantity = self.position
            if self.position > 0:
                trade_amount = sell_quantity * current_price
                percentage = (current_price - self.entry_price) / self.entry_price
            elif self.position < 0:
                trade_amount = sell_quantity * current_price * -1
                percentage = (self.entry_price - current_price) / current_price

            # 거래 수수료 계산 (테이커 비용)
            fee = trade_amount * self.taker_fee_pct

            # 포지션 업데이트
            self.position = 0

            # 계좌 잔고 업데이트
            self.account_balance += trade_amount - fee

            logger.info(f"청산 주문: 가격={current_price}, 수량={sell_quantity}, 수수료={fee}, 잔고={self.account_balance:.2f}, 포지션={self.position}, 이익={percentage * self.leverage:.2%}")
        else:
            logger.info("현재 보유 중인 포지션이 없습니다. 청산 주문을 실행할 수 없습니다.")

    def _get_reward(self) -> float:
        """현재 상태에 대한 보상을 계산합니다."""
        if self.position == 0:
            current_balance = self.account_balance
            return_rate = (current_balance - self.initial_account_balance) / self.initial_account_balance
            reward = return_rate * 100
            self.initial_account_balance = current_balance
        else:
            # 포지션이 청산되지 않았으므로 보상은 0
            reward = 0.0

        return reward

# 유닛 테스트 추가
import unittest

class TestFuturesTradingEnv(unittest.TestCase):
    def setUp(self):
        self.env = FuturesTradingEnv()

    def test_reset(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)

    def test_step(self):
        obs = self.env.reset()
        action = Action.BUY.value
        next_obs, reward, done, info = self.env.step(action)
        self.assertIsInstance(next_obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_invalid_action(self):
        with self.assertRaises(ValueError):
            self.env.step(len(Action))

if __name__ == '__main__':
    unittest.main() 
