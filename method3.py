# 모듈 임포트
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from LSTM import LSTM
import os

# 시계열 데이터셋 생성 함수
def create_dataset(X, y, time_step=1):
    dataX, dataY = [], []
    for i in range(len(X)-time_step-1):
        a = X[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(y[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# 현재 작업 디렉토리 설정
current_path = os.getcwd()
if not os.path.exists(os.path.join(current_path, 'method3')):
    os.mkdir(os.path.join(current_path, 'method3'))
os.chdir(os.path.join(current_path, 'method3'))

# 삼성 및 SK하이닉스 주식 데이터 다운로드
start_date = '2014-01-01'
end_date = '2024-01-01'

sk_hynix_data = yf.download('000660.KS', start=start_date, end=end_date, progress=False)
samsung_data = yf.download('005930.KS', start=start_date, end=end_date, progress=False)

sk_hynix_data = sk_hynix_data[['Close']]
samsung_data = samsung_data[['Close']]

# 폰트 설정
fontdict = {'fontsize': 12, 'weight': 'bold'}

# 데이터 시각화
plt.figure(figsize=(14,5))
plt.plot(sk_hynix_data, label='SK Hynix')
plt.plot(samsung_data, label='Samsung')
plt.title('Stock Prices', fontdict=fontdict)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('stock_prices.png')

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_sk_hynix = scaler.fit_transform(sk_hynix_data)
scaled_samsung = scaler.fit_transform(samsung_data)

# LSTM 모델을 위한 데이터셋 생성
time_step = 100
X, y = create_dataset(scaled_sk_hynix, scaled_samsung, time_step)

# 데이터를 학습 및 테스트 세트로 분리
train_size = int(len(X) * 0.65)
test_size = len(X) - train_size

X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# LSTM 입력 형식에 맞게 데이터 형태 변환
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 모델 생성 및 로드
model = LSTM()
model.load_state_dict(torch.load(os.path.join(current_path, 'model_state_dict.pt')))

# 모델 예측
model.eval()

test_inputs = X_train[-time_step:].tolist()
predictions = []

for i in range(len(X_test)):
    seq = torch.FloatTensor(test_inputs[-time_step:])
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        predictions.append(model(seq.view(-1, 1, 1)).item())
    test_inputs.append(X_test[i].tolist())

# 예측값 역정규화
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 실제값과 예측값 비교 시각화
plt.figure(figsize=(14,5))
plt.plot(samsung_data.index[len(samsung_data) - len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual')
plt.plot(samsung_data.index[len(samsung_data) - len(y_test):], predictions, label='Predicted')
plt.title('Samsung Stock Price Prediction by SK Hynix Stock Price', fontdict=fontdict)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.savefig('stock_price_prediction.png')

# 실제값과 예측값 평균변화율 계산 및 데이터프레임 생성
predictions_df = pd.DataFrame(predictions, index=samsung_data.index[len(samsung_data) - len(y_test):], columns=['Predicted'])
predictions_df['diff'] = predictions_df.diff()
actual_df = pd.DataFrame(scaler.inverse_transform(y_test.reshape(-1, 1)), index=samsung_data.index[len(samsung_data) - len(y_test):], columns=['Actual'])
actual_df['diff'] = actual_df.diff()

# dropna
predictions_df = predictions_df.drop(predictions_df.index[0])
actual_df = actual_df.drop(actual_df.index[0])

# 예측값과 실제값의 증가 및 감소 시각화
fig, axs = plt.subplots(2,1, figsize=(14,10))
axs[0].plot(predictions_df['Predicted'], label='Predicted', color='r', alpha=0)
for i in range(len(predictions_df)-1):
    if predictions_df.iloc[i+1, 1] > 0:
        axs[0].plot(predictions_df.iloc[i:i+2, 0], color='r')
    else:
        axs[0].plot(predictions_df.iloc[i:i+2, 0], color='b')
axs[0].set_title('Predicted', fontdict=fontdict)
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Price')
axs[0].legend(loc='upper left')

axs[1].plot(actual_df['Actual'], label='Actual', color='w', alpha=0)
for i in range(len(actual_df)-1):
    if actual_df.iloc[i+1, 1] > 0:
        axs[1].plot(actual_df.iloc[i:i+2, 0], color='r')
    else:
        axs[1].plot(actual_df.iloc[i:i+2, 0], color='b')
axs[1].set_title('Actual', fontdict=fontdict)
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Price')
axs[1].legend(loc='upper left')

plt.tight_layout()
plt.savefig('stock_price_prediction_inc_dec.png')

# 예측값과 실제값의 증가 및 감소 비교
check = pd.DataFrame(columns=['correct', 'wrong'], index=predictions_df.index)
cnt = 0
for i in range(len(predictions_df['diff'])):
    if predictions_df.iloc[i, 1] * actual_df.iloc[i, 1] > 0:
        check.iloc[i, 0] = 1
        cnt += 1
    else:
        check.iloc[i, 1] = -1

# 예측한 증감대로 수익률 계산
start_idx_p = 0
start_idx_a = 0
start_price_p = 0
start_price_a = 0
tmp_predict = 0
tmp_actual = 0
flag_p = 0
flag_a = 0
for i in range(len(predictions_df)-1):
    start_idx_p += 1
    if predictions_df.iloc[i+1, 1] < 0:
        continue
    else:
        start_price_p = predictions_df.iloc[i, 0]
        break
for i in range(len(actual_df)-1):
    start_idx_a += 1
    if actual_df.iloc[i+1, 1] < 0:
        continue
    else:
        start_price_a = actual_df.iloc[i, 0]
        break
revenue_p = -start_price_p
revenue_a = -start_price_a

# 극소 극대 판별법으로 저점매수 고점매도
for i in range(1, len(predictions_df)-1):
    if predictions_df.iloc[i, 1] <= 0 and predictions_df.iloc[i+1, 1] >= 0:
        revenue_p -= actual_df.iloc[i, 0]
        flag_p = 1
    elif predictions_df.iloc[i+1, 1] <= 0 and predictions_df.iloc[i, 1] >= 0:
        revenue_p += actual_df.iloc[i, 0]
        tmp_predict = revenue_p
        flag_p = 0
    
    if actual_df.iloc[i, 1] <= 0 and actual_df.iloc[i+1, 1] >= 0:
        revenue_a -= actual_df.iloc[i, 0]
        flag_a = 1
    elif actual_df.iloc[i, 1] >= 0 and actual_df.iloc[i+1, 1] <= 0:
        revenue_a += actual_df.iloc[i, 0]
        tmp_actual = revenue_a
        flag_a = 0

if flag_p:
    revenue_p = tmp_predict

if flag_a:
    revenue_a = tmp_actual

fig = plt.figure(figsize=(8,8))
plt.pie([cnt, len(predictions_df['diff'])-cnt], labels=['correct', 'wrong'], autopct='%1.1f%%', startangle=-60,
        colors=['#8fd9b6', '#ff9999'], explode=(0.05, 0))
plt.text(0.5, -1.3, f'correct rate:{cnt/len(check):.2f}\nPredicted Start Price: {start_price_p:.2f}\nPredicted Revenue: {revenue_p:.2f}\nPredicted Revenue Rate: {(revenue_p+start_price_p)/start_price_p:.2f}\nActual Start Price: {start_price_a:.2f}\nActual Revenue: {revenue_a:.2f}\nActual Revenue Rate: {(revenue_a+start_price_a)/start_price_a:.2f}\n')
plt.title('Prediction Accuracy', fontdict=fontdict)
plt.tight_layout()
plt.savefig('prediction_accuracy.png')

# 현재 작업 디렉토리 복귀
os.chdir(current_path)