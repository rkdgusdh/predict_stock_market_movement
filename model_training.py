import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from LSTM import LSTM

# 시계열 데이터셋 생성 함수
def create_dataset(X, y, time_step=1):
    dataX, dataY = [], []
    for i in range(len(X)-time_step-1):
        a = X[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(y[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# 삼성 및 SK하이닉스 주식 데이터 다운로드
start_date = '2014-01-01'
end_date = '2024-01-01'

sk_hynix_data = yf.download('000660.KS', start=start_date, end=end_date, progress=False)
samsung_data = yf.download('005930.KS', start=start_date, end=end_date, progress=False)

sk_hynix_data = sk_hynix_data[['Close']]
samsung_data = samsung_data[['Close']]

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

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 1000

for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(torch.FloatTensor(seq))

        single_loss = loss_function(y_pred, torch.FloatTensor([labels]))
        single_loss.backward()
        optimizer.step()

    if i%100 == 0:
        print(f'Epoch {i} loss: {single_loss.item()}')

# 모델 저장
torch.save(model.state_dict(), 'model_state_dict.pt')