import pandas as pd


def simple_moving_average(data, window):
    sma = data.rolling(window=window).mean()
    return sma


def test_simple_moving_average():
    data = pd.Series([1, 2, 3, 4, 5])
    window = 2
    expected_result = pd.Series([None, 1.5, 2.5, 3.5, 4.5])
    result = simple_moving_average(data, window)
    assert result.equals(expected_result), f'Expected {expected_result}, but got {result}'


def test_buy_and_sell(data):
    df = pd.DataFrame(data)
    start_idx = 0
    start_price = 0
    for i in range(len(df)-1):
        start_idx += 1
        if df.iloc[i+1, 1] < 0:
            continue
        else:
            start_price = df.iloc[i, 0]
            break
    revenue = -start_price
    tmp = 0
    flag = 0

    for i in range(start_idx, len(df)-1):
        if df.iloc[i, 1] <= 0 and df.iloc[i+1, 1] >= 0:
            revenue -= df.iloc[i, 0]
            flag = 1
        elif df.iloc[i, 1] >= 0 and df.iloc[i+1, 1] <= 0:
            revenue += df.iloc[i, 0]
            tmp = revenue
            flag = 0

    if flag:
        revenue = tmp
    
    if start_price == 0:
        return None, None
    return start_price, revenue


test_simple_moving_average()

data1 = {
    'price': [100, 95, 90, 85, 80, 85, 90, 80, 100, 50],
    'diff': [0, -1, -1, -1, -1, 1, 1, -1, 1, -1]
}

data2 = {
    'price': [100, 95, 90, 85, 80, 85, 90, 80, 100, 50, 60],
    'diff': [0, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1]
}

data3 = {
    'price': [10, 20, 10],
    'diff': [0, 1, -1]
}

data4 = {
    'price': [40, 30, 20, 10],
    'diff': [0, -1, -1, -1]
}

# 시작이 고점일떄
# 최종 flag=0 일때
start_price, revenue = test_buy_and_sell(data1)
assert start_price == 80, f'Expected {80}, but got {start_price}'
assert revenue == -80+90-80+100, f'Expected {-80+90-80+100}, but got {revenue}'

# 최종 flag=1 일때
start_price, revenue = test_buy_and_sell(data2)
assert start_price == 80, f'Expected {80}, but got {start_price}'
assert revenue == -80+90-80+100, f'Expected {-80+90-80+100}, but got {revenue}'

# 시작이 저점일때
start_price, revenue = test_buy_and_sell(data3)
assert start_price == 10, f'Expected {10}, but got {start_price}'
assert revenue == -10+20, f'Expected {10}, but got {revenue}'

# 가격이 계속 내려갈때
start_price, revenue = test_buy_and_sell(data4)
assert start_price == None, f'Expected {None}, but got {start_price}'
assert revenue == None, f'Expected {None}, but got {revenue}'

print('All tests passed')