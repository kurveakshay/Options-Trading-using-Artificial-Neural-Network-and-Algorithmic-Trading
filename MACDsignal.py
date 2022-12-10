#Data Source
!pip install yfinance
import yfinance as yf

#Data viz
import plotly.graph_objs as go
import plotly as pt
!pip install pandas_ta
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import date
import numpy as np
import pandas as pd
import pandas_datareader.data as webd


data = yf.download(tickers='^NSEI', period='1d', interval='1m')
macd = ta.macd(data['Close'])

## Now that we've got the MACD columns, we'll merge them with our main dataframe.
data = pd.concat([data, macd], axis=1).reindex(data.index)

def MACD_Strategy(df, risk):
    MACD_Buy=[]
    MACD_Sell=[]
    position=False

    for i in range(0, len(df)):
        if df['MACD_12_26_9'][i] > df['MACDs_12_26_9'][i] :
            MACD_Sell.append(np.nan)
            if position ==False:
                MACD_Buy.append(df['Close'][i])
                position=True
            else:
                MACD_Buy.append(np.nan)
        elif df['MACD_12_26_9'][i] < df['MACDs_12_26_9'][i] :
            MACD_Buy.append(np.nan)
            if position == True:
                MACD_Sell.append(df['Close'][i])
                position=False
            else:
                MACD_Sell.append(np.nan)
        elif position == True and df['Close'][i] < MACD_Buy[-1] * (1 - risk):
            MACD_Sell.append(df["Close"][i])
            MACD_Buy.append(np.nan)
            position = False
        elif position == True and df['Close'][i] < df['Close'][i - 1] * (1 - risk):
            MACD_Sell.append(df["Close"][i])
            MACD_Buy.append(np.nan)
            position = False
        else:
            MACD_Buy.append(np.nan)
            MACD_Sell.append(np.nan)

    data['MACD_Buy_Signal_price'] = MACD_Buy
    data['MACD_Sell_Signal_price'] = MACD_Sell

 ## Included a Risk factor in our strategy. If the current price goes beyond the acceptable limits
 ## or if the current price goes beyond the specified percentage, we'll sell.
 ##In simple terms, we've added a STOP LOSS and a Trailing Stop loss to our strategy.
MACD_strategy = MACD_Strategy(data, 0.025)

def MACD_color(data):
    MACD_color = []
    for i in range(0, len(data)):
        if data['MACDh_12_26_9'][i] > data['MACDh_12_26_9'][i - 1]:
            MACD_color.append(True)
        else:
            MACD_color.append(False)
    return MACD_color

data['positive'] = MACD_color(data)

## plot
plt.rcParams.update({'font.size': 10})
fig, ax1 = plt.subplots(figsize=(14,8))
fig.suptitle('NIFTY50', fontsize=10, backgroundcolor='blue', color='white')
ax1 = plt.subplot2grid((14, 8), (0, 0), rowspan=8, colspan=14)
ax1.set_ylabel('Price in ₨')
ax1.plot('Close',data=data, label='Close Price', linewidth=0.5, color='blue')
ax1.scatter(data.index, data['MACD_Buy_Signal_price'], color='green', marker='^', alpha=1)
ax1.scatter(data.index, data['MACD_Sell_Signal_price'], color='red', marker='v', alpha=1)
ax1.legend()
ax1.grid()
ax1.set_xlabel('Date', fontsize=8)


