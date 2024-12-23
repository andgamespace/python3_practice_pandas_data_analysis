import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_data(tickers, interval, filename_prefix="data"):
    """
    Downloads historical data for the given tickers from Yahoo Finance and saves to CSV files.

    Parameters:
        tickers (list): List of company tickers (e.g., ['AAPL', 'MSFT']).
        interval (str): The data interval (e.g., '1m', '1h', '1d', etc.).
            Available options:
                '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        filename_prefix (str): Prefix for the CSV filenames (default is "data").

    Returns:
        None
    """
    # Set the date range based on the interval
    if interval == "1m":
        start_date = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
    elif interval in ["2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
        start_date = (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d')
    else:
        start_date = "2015-01-01"  # Default start date for longer intervals

    # Loop through each ticker
    for ticker in tickers:
        try:
            print(f"Downloading data for {ticker} with interval '{interval}'...")
            # Fetch historical data
            data = yf.download(ticker, interval=interval, start=start_date, end=None)

            # Save to CSV
            filename = f"{filename_prefix}_{ticker}_{interval}.csv"
            data.to_csv(filename, encoding="utf-8")

            print(f"Data saved to {filename}\n")
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

# Example usage
tickers = ["AAPL", "GOOGL"]  
# List of company tickers
#"MSFT", "GOOGL", "AMZN", "TSLA"]
# Data interval (choose one of the options):
# '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
interval = "15m"  
download_data(tickers, interval)

           


