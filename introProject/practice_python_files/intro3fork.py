import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from bokeh.models import DatetimeTickFormatter

# Calculate dates
end_date = datetime.now()
start_date = end_date - timedelta(days=59)

# Fetch 15-minute data
aapl_data = yf.Ticker("AAPL")
aapl_df = aapl_data.history(
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval='15m'
)
aapl_df = aapl_df[['Open', 'High', 'Low', 'Close', 'Volume']]

def SMA(array, window):
    """Calculate Simple Moving Average that matches input length"""
    array = np.array(array)
    output = np.zeros_like(array) + np.nan
    for i in range(window - 1, len(array)):
        output[i] = np.mean(array[i - window + 1:i + 1])
    return output

class SmaCross(Strategy):
    n1 = 20  # fast MA (5 hours of trading)
    n2 = 40  # slow MA (10 hours of trading)
    
    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)
    
    def next(self):
        if not np.isnan(self.sma1[-1]) and not np.isnan(self.sma2[-1]):
            # Only trade if there's sufficient volume
            if self.data.Volume[-1] > self.data.Volume[-20:].mean():
                if self.sma1[-1] > self.sma2[-1] and self.sma1[-2] <= self.sma2[-2]:
                    self.position.close()
                    self.buy()
                elif self.sma1[-1] < self.sma2[-1] and self.sma1[-2] >= self.sma2[-2]:
                    self.position.close()
                    self.sell()

# Run backtest
bt = Backtest(aapl_df, SmaCross, 
              cash=10000, 
              commission=.002,
              exclusive_orders=True)
stats = bt.run()
print(stats)
print(f"\nFinal Portfolio Value: ${stats['Equity Final [$]']:.2f}")
print(f"Return: {stats['Return [%]']:.2f}%")
print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
print(f"Number of Trades: {stats['# Trades']}")
print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")

# Plot the backtest results
bt.plot()

# Plot the SMA crossover
plt.figure(figsize=(12, 6))
plt.plot(aapl_df.index, aapl_df['Close'], label='Close Price')
plt.plot(aapl_df.index, SMA(aapl_df['Close'], 20), label='20-period SMA')
plt.plot(aapl_df.index, SMA(aapl_df['Close'], 40), label='40-period SMA')
plt.legend(loc='best')
plt.title('SMA Crossover Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()