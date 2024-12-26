import backtrader as bt
import datetime
from datetime import timedelta
import yfinance as yf
from strategies.strategy001 import TestStrategy
import pandas as pd 

cerebro = bt.Cerebro()

cerebro.broker.set_cash(1000000)

# Calculate the end date as “now” and start date as 59 days ago
end_date = datetime.datetime.now()
start_date = end_date - timedelta(days=59)

# Download data using yfinance with interval='2m'
df = yf.download('AAPL', interval='2m', start=start_date, end=end_date)
if df.empty:
    raise ValueError("No data returned from yfinance")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

# Reset index to have 'Date' as a column
df.reset_index(inplace=True)

# Rename columns to lowercase
df.columns = [col.lower() for col in df.columns]
if 'adj close' in df.columns:
    df.rename(columns={'adj close': 'close'}, inplace=True)

mapping = {}
for col in df.columns:
    if col.endswith('_aapl'):
        mapping[col] = col.replace('_aapl', '')
if 'datetime' in mapping.values():
    # rename 'datetime' to 'date'
    for k, v in mapping.items():
        if v == 'datetime':
            mapping[k] = 'date'
df.rename(columns=mapping, inplace=True)

if 'datetime' in df.columns:
    df.rename(columns={'datetime': 'date'}, inplace=True)

print("Final Columns:", df.columns)
if 'close' not in df.columns:
    for col in df.columns:
        if 'close' in col:
            df.rename(columns={col: 'close'}, inplace=True)

data = bt.feeds.PandasData(
    dataname=df,
    datetime='date',
    open='open',
    high='high',
    low='low',
    close='close',
    volume='volume',
    openinterest=None
)
cerebro.adddata(data)
cerebro.addstrategy(TestStrategy)
cerebro.addsizer(bt.sizers.FixedSize, stake=1000)
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.plot()