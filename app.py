import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pybit.unified_trading import HTTP
import os

# API från env vars
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')
TESTNET = True

if not API_KEY or not API_SECRET:
    print("Saknar API-nycklar")
    exit()

session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET)

SYMBOL = 'BTCUSDT'
INTERVAL = '5'
LIMIT = 200
TRAIN_LIMIT = 1000
MAX_QTY = 0.01
RISK_PERCENT = 0.01
SL_PERCENT = 0.01
TP_PERCENT = 0.02

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        rsi = np.full_like(prices, 100)
    else:
        rs = avg_gain / avg_loss
        rsi = np.full_like(prices, 100 - (100 / (1 + rs)))
    
    for i in range(period, len(prices)):
        current_gain = gains[i-1]
        current_loss = losses[i-1]
        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs)))
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    def ema(data, period):
        ema = np.zeros_like(data)
        sma = np.mean(data[:period])
        ema[:period] = sma
        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# Enkel Torch-modell
class SimplePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)  # Input: RSI, MACD, Histogram
        self.fc2 = nn.Linear(10, 1)  # Output: -1 till 1 för sell/hold/buy
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

model = SimplePredictor()

# Träna modellen
def train_model():
    response = session.get_kline(
        category='linear',
        symbol=SYMBOL,
        interval=INTERVAL,
        limit=TRAIN_LIMIT
    )
    if response['retCode'] != 0:
        print(f"Fel vid träningsdata: {response['retMsg']}")
        return

    klines = response['result']['list']
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['close'] = df['close'].astype(float)
    closes = df['close'].values[::-1]
    
    rsi = calculate_rsi(closes)
    macd, signal, hist = calculate_macd(closes)
    
    inputs = []
    labels = []
    for i in range(len(closes) - 1):
        inputs.append([rsi[i] / 100, macd[i], hist[i])
        ret = (closes[i+1] - closes[i]) / closes[i]
        labels.append(max(min(ret, 1), -1))
    
    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    print("Modell tränad!")

def get_balance():
    response = session.get_wallet_balance(accountType='UNIFIED', coin='USDT')
    if response['retCode'] == 0:
        for b in response['result']['list'][0]['coin']:
            if b['coin'] == 'USDT':
                return float(b['walletBalance'])
    return 0

def get_position():
    response = session.get_positions(category='linear', symbol=SYMBOL)
    if response['retCode'] == 0:
        for pos in response['result']['list']:
            if pos['symbol'] == SYMBOL and float(pos['size']) > 0:
                return float(pos['size']), pos['side'], float(pos['avgPrice'])
    return 0, None, None

def place_order(side, qty, price=None, entry_price=None, is_open=True):
    order_type = 'Market' if price is None else 'Limit'
    params = {
        'category': 'linear',
        'symbol': SYMBOL,
        'side': side,
        'orderType': order_type,
        'qty': str(qty),
        'timeInForce': 'GTC'
    }
    if price:
        params['price'] = str(price)
    
    if is_open and entry_price:
        if side == 'Buy':
            sl_price = entry_price * (1 - SL_PERCENT)
            tp_price = entry_price * (1 + TP_PERCENT)
        elif side == 'Sell':
            sl_price = entry_price * (1 + SL_PERCENT)
            tp_price = entry_price * (1 - TP_PERCENT)
        params['stopLoss'] = str(round(sl_price, 2))
        params['takeProfit'] = str(round(tp_price, 2))
        params['slTriggerBy'] = 'LastPrice'
        params['tpTriggerBy'] = 'LastPrice'
    
    response = session.place_order(**params)
    if response['retCode'] != 0:
        print(f"Fel vid order: {response['retMsg']}")
    return response

train_model()  # Träna en gång

while True:
    response = session.get_kline(
        category='linear',
        symbol=SYMBOL,
        interval=INTERVAL,
        limit=LIMIT
    )
    if response['retCode'] != 0:
        print(f"Fel vid data: {response['retMsg']}")
        time.sleep(60)
        continue

    klines = response['result']['list']
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['close'] = df['close'].astype(float)
    closes = df['close'].values[::-1]
    current_price = closes[-1]
    
    rsi = calculate_rsi(closes)[-1]
    macd, signal, hist = calculate_macd(closes)
    macd_val = macd[-1]
    hist_val = hist[-1]
    
    inputs = torch.tensor([[rsi / 100, macd_val, hist_val]], dtype=torch.float32)
    prediction = model(inputs).item()
    
    position_size, position_side, entry_price = get_position()
    balance = get_balance()
    
    sl_distance = current_price * SL_PERCENT
    risk_qty = (balance * RISK_PERCENT) / sl_distance if sl_distance > 0 else MAX_QTY
    qty = min(risk_qty, MAX_QTY)
    
    if position_size == 0:
        if rsi < 30 and prediction > 0.5 and qty > 0:
            print(f"Köpsignal (long)! RSI: {rsi}, Pred: {prediction}")
            place_order('Buy', qty, entry_price=current_price, is_open=True)
        elif rsi > 70 and prediction < -0.5 and qty > 0:
            print(f"Säljsignal (short)! RSI: {rsi}, Pred: {prediction}")
            place_order('Sell', qty, entry_price=current_price, is_open=True)
    else:
        if position_side == 'Buy' and rsi > 70 and prediction < -0.5:
            print(f"Stäng long! RSI: {rsi}, Pred: {prediction}")
            place_order('Sell', position_size, is_open=False)
        elif position_side == 'Sell' and rsi < 30 and prediction > 0.5:
            print(f"Stäng short! RSI: {rsi}, Pred: {prediction}")
            place_order('Buy', position_size, is_open=False)
    
    print(f"Väntar... RSI: {rsi}, MACD Hist: {hist_val}, Pred: {prediction}, Balans: {balance}")
    time.sleep(60 * 5)