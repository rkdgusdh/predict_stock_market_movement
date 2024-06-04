# 모듈 임포트
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

# 현재 작업 디렉토리 설정
current_path = os.getcwd()
if not os.path.exists(os.path.join(current_path, 'method2')):
    os.mkdir(os.path.join(current_path, 'method2'))
os.chdir(os.path.join(current_path, 'method2'))

# 주식 데이터 다운로드
start_date = '2014-01-01'
end_date = '2024-01-01'
apple = yf.download('AAPL', start=start_date, end=end_date, progress=False)
samsung = yf.download('005930.KS', start=start_date, end=end_date, progress=False)

# 애플과 삼성의 수익률에 대한 상관관계 시각화
apple_close = apple['Close'] * 1350 # 환율 1350원 기준으로 환산(시각화를 위해)
samsung_close = samsung['Close']

# 일간 수익률 계산
apple_returns = apple_close.pct_change().dropna().rename('Returns_AAPL')
samsung_returns = samsung_close.pct_change().dropna().rename('Returns_SSNLF')

# 데이터 병합
merged_data = pd.merge(apple_returns, samsung_returns, left_index=True, right_index=True)

# 상관관계 계산(피어슨 상관계수)
correlation = merged_data.corr().iloc[0, 1]

# 폰트 설정
fontdict = {'fontsize': 12, 'weight': 'bold'}

# 수익률 산점도 시각화
plt.figure(figsize=(12, 8))
plt.scatter(merged_data['Returns_AAPL'], merged_data['Returns_SSNLF'], alpha=0.6)
plt.title('Scatter plot of Apple vs Samsung Daily Returns', fontdict=fontdict)
plt.xlabel('Apple Daily Returns')
plt.ylabel('Samsung Daily Returns')
plt.yticks([-0.1, -0.05, 0, 0.05, 0.1])
plt.text(x=0.01, y=-0.12, s=f"Correlation between Apple and Samsung stock returns: {correlation:.2f}", fontsize=10)
plt.legend(['AAPL vs SSNLF'])
plt.grid()
plt.tight_layout()
plt.savefig('apple_vs_samsung_returns.png')

# 히트맵 시각화를 이용해 여러 주식 간의 상관관계 분석
tickers = {
    'Samsung': '005930.KS',
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Amazon': 'AMZN',
    'Intel' : 'INTC',
    'Google': 'GOOGL',
    'Nvidia': 'NVDA',
    'Naver': '035420.KS',
    'Kakao': '035720.KS',
    'SK_Hynix': '000660.KS'
}

start_date = '2014-01-01'
end_date = '2024-01-01'

data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data[name] = df['Close']

prices = pd.DataFrame(data)
returns = prices.pct_change().dropna()

correlation_matrix = returns.corr()

# 여러 주식간 상관관계 히트맵 시각화
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Reds')
plt.xticks(rotation=0)
plt.title('Correlation Matrix', fontdict=fontdict)
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# 선형회귀 분석을 통해 삼성전자와 SK하이닉스 주식의 상관관계 분석
samsung_returns = returns['Samsung']
skhynix_returns = returns['SK_Hynix']

correlation = correlation_matrix.loc['Samsung', 'SK_Hynix']

X = samsung_returns.values.reshape(-1, 1)
y = skhynix_returns.values

model = LinearRegression()
model.fit(X, y)

# 수익률 산점도 시각화
plt.figure(figsize=(8, 6))
plt.scatter(samsung_returns, skhynix_returns, alpha=0.5)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.text(x=0.055, y=0.08, s=f"Correlation: {correlation:.2f}", fontsize=12)
plt.title('Samsung vs SK Hynix Daily Returns', fontdict=fontdict)
plt.xlabel('Samsung Daily Returns')
plt.ylabel('SK Hynix Daily Returns')
plt.grid()
plt.legend(['Returns', 'Linear Regression'])
plt.tight_layout()
plt.savefig('samsung_vs_skhynix_returns.png')

# 현재 작업 디렉토리 복구
os.chdir(current_path)