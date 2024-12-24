from sklearn.linear_model import LinearRegression  # Import LinearRegression
from sklearn import model_selection
from sklearn import preprocessing
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from matplotlib import style
from sklearn.svm import SVR  # Import SVR
import pickle

style.use('ggplot')


googl_data = yf.Ticker("GOOGL")
googl_df = googl_data.history(period="1y")
print(googl_df.head())

# Calculate the start date (59 days ago)
end_date = datetime.now()
start_date = end_date - timedelta(days=59)

# Fetch the 2-minute data
googl_2m_data = yf.download("GOOGL", interval='2m', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# Store in DataFrame
googl_2m_df = pd.DataFrame(googl_2m_data)

# Print the 2-minute DataFrame
print(googl_2m_df.head())


# Reformatting the DataFrame
df = googl_df[['Open', 'High', 'Low', 'Close', 'Volume']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

# Define the forecast column
forecast_col = 'Close'
df.fillna(value=-99999, inplace=True)

# Change forecast_out to predict the next 14 days
forecast_out = 14
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Use LinearRegression instead of SVR
clf = LinearRegression()
clf.fit(X_train, y_train)
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)


forecast_set = clf.predict(X_lately)
accuracy = clf.score(X_test, y_test)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

# Print accuracy before plotting
print(f"Accuracy for GOOGL: {accuracy}")

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()

one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# Ensure the index is datetime without timezone
df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

# Plot the data
df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Change the ticker symbol to Apple (AAPL)
aapl_data = yf.Ticker("AAPL")
aapl_df = aapl_data.history(period="1y")
print(aapl_df.head())

# Calculate the start date (59 days ago)
end_date = datetime.now()
start_date = end_date - timedelta(days=59)

# Fetch the 2-minute data
aapl_2m_data = yf.download("AAPL", interval='2m', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# Store in DataFrame
aapl_2m_df = pd.DataFrame(aapl_2m_data)

# Print the 2-minute DataFrame
print(aapl_2m_df.head())

# Reformatting the DataFrame
df = aapl_df[['Open', 'High', 'Low', 'Close', 'Volume']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

# Define the forecast column
forecast_col = 'Close'
df.fillna(value=-99999, inplace=True)

# Change forecast_out to predict the next 14 days
forecast_out = 14
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Use LinearRegression instead of SVR
clf = LinearRegression()
clf.fit(X_train, y_train)
forecast_set = clf.predict(X_lately)
accuracy = clf.score(X_test, y_test)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

# Print accuracy before plotting
print(f"Accuracy for AAPL: {accuracy}")

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()

one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# Ensure the index is datetime without timezone
df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

# Plot the data
df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()