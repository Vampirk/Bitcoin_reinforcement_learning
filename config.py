# config.py

# 데이터 경로
DATA_PATH = 'processed_data.csv'

# 환경 설정
INITIAL_ACCOUNT_BALANCE = 10000  # 초기 계좌 잔고

# 거래 수수료 설정
MAKER_FEE_PCT = 0.0002  # 메이커 수수료율 (0.02%)
TAKER_FEE_PCT = 0.0006  # 테이커 수수료율 (0.06%)

# 레버리지 설정
LEVERAGE = 10  # 고정된 레버리지 값

# 에이전트 설정
AGENT_TYPE = 'dqn'  # 에이전트 유형 (dqn, ddpg, ppo 등)
GAMMA = 0.99  # 할인율
BATCH_SIZE = 32  # 배치 크기
BUFFER_SIZE = 10000  # 리플레이 버퍼 크기
LEARNING_RATE = 0.001  # 학습률
UPDATE_EVERY = 4  # 타깃 네트워크 업데이트 주기

# 신경망 설정
NEURAL_NET_CONFIG = {
    'input_size': 27,  # 입력 크기 (가격, 기술 지표 수 등)
    'hidden_sizes': [64, 32],  # 은닉층 크기
    'output_size': 3  # 출력 크기 (매수, 매도, 청산, 유지)
}

# 에피소드 설정
NUM_EPISODES = 1000  # 총 에피소드 수
MAX_EPISODE_LEN = 1000  # 최대 에피소드 길이

# 기타 설정
SEED = 42  # 랜덤 시드
