import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define and download data from yahoo finance
ticker_symbol = "SPY"
start_date = "2015-1-1"
end_date = "2025-5-30"

# Get Closing Price from data
data = yf.download(ticker_symbol, start = start_date, end = end_date)[['Close']]

# Section 2, calculate the moving averages

# Define the windows
short_window = 10
long_window = 50

# Calculate the moving averages
data['SMA10'] = data['Close'].rolling(window=short_window).mean()
data['SMA50'] = data['Close'].rolling(window=long_window).mean()



# Section3, Generate Trading Signal

#This line tells the computer when to go in and out of the market. We go in (1) when the 10-day moving average is above
# the 50-day moving average. We leave when its not (0)
data['Position'] = np.where(data['SMA10'] > data['SMA50'], 1, 0)

#Position Change
data['Signal'] = data['Position'].diff()
# print(data[data['Signal'] !=0].tail())

# Section 4, calculate strat returns

#Calculate Benchmark
data['Returns'] = data['Close'].pct_change()

# Calculate strategy returns
# Multiple the daily asset return from the previous day, we shift it to prevent lookback bias
data['Strategy_Returns'] = data['Returns'] * data['Position'].shift (1)

# Section 5 & 6
data = data.dropna()

data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()

# Performance Metrics, drawdown and Sharpe

data['Strategy_Peak'] = data['Cumulative_Strategy'].cummax()
data['Strategy_Drawdown'] = (data['Cumulative_Strategy'] - data['Strategy_Peak']) / data['Strategy_Peak']
max_drawdown = data['Strategy_Drawdown'].min()

#Sharpe ratio

strategy_std_dev = data['Strategy_Returns'].std()
if strategy_std_dev != 0:
    strategy_sharpe_ratio = data['Strategy_Returns'].mean() / strategy_std_dev * (252**0.5)
else:
    strategy_sharpe_ratio = 0

# Benchmark

benchmark_std_dev = data['Returns'].std()
if benchmark_std_dev != 0:
    benchmark_sharpe_ratio = data['Returns'].mean() / benchmark_std_dev * (252**0.5)
else:
    benchmark_sharpe_ratio = 0

# --- Simple Win Rate Calculation ---
print("\nCalculating Win Rate...")

# Find all buy and sell signals
buy_signals = data[data['Signal'] == 1]['Close'].values
sell_signals = data[data['Signal'] == -1]['Close'].values

# Match buys with sells (assuming they alternate properly)
num_trades = min(len(buy_signals), len(sell_signals))

if num_trades > 0:
    # Calculate returns for each trade
    returns = (sell_signals[:num_trades] - buy_signals[:num_trades]) / buy_signals[:num_trades]

    # Count winning trades
    winning_trades = (returns > 0).sum()

    # Calculate win rate
    win_rate = (winning_trades / num_trades) * 100

else:
    win_rate = 0
    print("No trades found.")


# Number of trades

initial_capital = 2000

strategy_final_value = initial_capital * data['Cumulative_Strategy'].iloc[-1]
benchmark_final_value = initial_capital * data['Cumulative_Returns'].iloc[-1]


num_trades = len(data[data['Signal'] != 0])

# Strategy Performance
print("\n--- STRATEGY PERFORMANCE ---")
# --- MODIFIED to include dollar amounts ---
print(f"Starting Capital: ${initial_capital:,.2f}")
print(f"Final Value: ${strategy_final_value:,.2f}")
print(f"Profit: ${strategy_final_value - initial_capital:,.2f}")
print(f"Cumulative Return: {(data['Cumulative_Strategy'].iloc[-1] - 1):.2%}")
print(f"Annualized Sharpe Ratio: {strategy_sharpe_ratio:.2f}") 
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Winning trades: {winning_trades}")
print(f"Win Rate: {win_rate:.2f}%")

# Benchmark Performance
print("\n--- BUY & HOLD (BENCHMARK) PERFORMANCE ---")
# --- MODIFIED to include dollar amounts ---
print(f"Starting Capital: ${initial_capital:,.2f}")
print(f"Final Value: ${benchmark_final_value:,.2f}")
print(f"Profit: ${benchmark_final_value - initial_capital:,.2f}")
print(f"Cumulative Return: {(data['Cumulative_Returns'].iloc[-1] - 1):.2%}")
print(f"Annualized Sharpe Ratio: {benchmark_sharpe_ratio:.2f}") 

print(f"\nTotal Trades Executed: {num_trades}")
print("-" * 50)

# Section 6, Plots and Visualization

plt.figure(figsize=(12, 12))

#Plot 1: Price, SMAs, Signals
plt.subplot(3, 1, 1)
plt.plot(data['Close'], label ='Close Price', alpha=0.6)
plt.plot(data['SMA10'], label='10-Day SMA', color ='orange', linestyle='--')
plt.plot(data['SMA50'], label= '50-Day SMA', color = 'purple', linestyle='--')

plt.plot(data[data['Signal'] == 1].index,
         data['SMA10'][data['Signal'] ==1],
         '^', markersize = 10, color = 'g', label ='Buy Signal')

plt.plot(data[data['Signal'] == -1].index,
         data['SMA10'][data['Signal'] == -1],
         '^', markersize = 10, color = 'r', label = 'Sell Signal')
plt.title(f'{ticker_symbol} Price, SMAs, and Crossover Signals')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

#Plot 2, curve

plt.subplot(3, 1, 2)
plt.plot(data['Cumulative_Strategy'], label='SMA Crossover Strategy', color='green')
plt.plot(data['Cumulative_Returns'], label='Buy & Hold Strategy', color = 'blue', alpha = 0.7)
plt.title('Strategy Performance vs. Buy & Hold')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)

#Plot 3, Drawdown

plt.subplot(3, 1, 3)
plt.plot(data['Strategy_Drawdown'], label= 'Strategy Drawdown', color = 'red')
plt.fill_between(data.index, data['Strategy_Drawdown'], 0, color = 'red', alpha = 0.3)
plt.title('Strategy Drawdown')
plt.ylabel('Drawdown')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
