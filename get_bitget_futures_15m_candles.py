import ccxt
import pandas as pd
from datetime import datetime, timedelta

def fetch_bitget_futures_data(symbol, timeframe, filename):
    # Bitget 거래소 객체 생성
    exchange = ccxt.bitget({
        'options': {
            'defaultType': 'future',  # 선물 거래를 위해 defaultType 설정
        }
    })

    # 시작 날짜와 종료 날짜 설정
    end_date = datetime.utcnow()  # 현재 UTC 시간
    start_date = end_date - timedelta(days=365)  # 1년 전

    end_timestamp = exchange.parse8601(end_date.isoformat() + 'Z')
    start_timestamp = exchange.parse8601(start_date.isoformat() + 'Z')

    # OHLCV 데이터 수집을 위한 빈 리스트 생성
    ohlcv_data = []

    # 총 데이터 수 계산
    total_data = int((end_timestamp - start_timestamp) / (exchange.parse_timeframe(timeframe) * 1000))

    # 반복문을 통해 데이터 수집
    collected_data = 0
    while True:
        try:
            # 1000개의 캔들 데이터 가져오기
            data = exchange.fetch_ohlcv(f'{symbol}', timeframe, start_timestamp)
            ohlcv_data += data
            collected_data += len(data)

            # 진행률 출력
            progress = (collected_data / total_data) * 100
            print(f"Progress: {progress:.2f}%")

            # 다음 날짜 범위 설정
            start_timestamp = data[-1][0] + exchange.parse_timeframe(timeframe) * 1000
            if start_timestamp >= end_timestamp:
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    # OHLCV 데이터를 Pandas DataFrame으로 변환
    cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(ohlcv_data, columns=cols)

    # 타임스탬프를 인덱스로 설정하고 날짜 형식으로 변환
    df = df.set_index('Timestamp')
    df.index = pd.to_datetime(df.index, unit='ms')

    # DataFrame을 CSV 파일로 저장
    df.to_csv(filename)

    return df

# 함수 사용 예시
symbol = 'ETH/USDT:USDT'
timeframe = '15m'
filename = 'bitget_futures_15m_candles.csv'

df = fetch_bitget_futures_data(symbol, timeframe, filename)
print(df)