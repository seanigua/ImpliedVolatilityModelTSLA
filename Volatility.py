import pandas as pd
import yfinance as yf
from scipy.constants import sigma

ticker = 'TSLA'
stock = yf.Ticker(ticker)

expiration_dates = stock.options
print(f"available expiration dates: {expiration_dates}")

exp_date = expiration_dates[0]
options_chain = stock.option_chain()

calls = options_chain.calls
puts = options_chain.puts

print(calls.head())

from scipy.optimize import brentq
from math import log, sqrt, exp
from scipy.stats import norm

def bs_call_price(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1-sigma*sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def implied_volatility (option_price, S, K, T, r):
    objective_function = lambda sigma: bs_call_price(S,K,T,r,sigma) - option_price
    return brentq(objective_function,1e-6,5)

S = 450
K = 455
T = 30/365
r = 0.01
option_price = 7.5

iv = implied_volatility(option_price, S, K, T, r)
print(f"implied volatility: {iv:.2%}")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

strikes = np.linspace(300, 400, 10)
maturities = np.linspace(0.05, 0.5, 10)

vol_surface = np.random.rand(10, 10) * 0.2 +0.1

X, Y = np.meshgrid(strikes, maturities)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, vol_surface, cmap='viridis')

ax.set_xlabel('Strike Price')
ax.set_ylabel('Maturity (Years)')
ax.set_zlabel('Implied Volatility')
ax.set_title('Volatility Surface for TSLA')

plt.show()

