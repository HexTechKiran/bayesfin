"""
Generates distributions for the volatility of stocks in the S&P 500.

Uses a 5 year dataset of daily stock prices for all stocks in the S&P 500
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.stats import gaussian_kde

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    df = pd.read_csv("all_stocks_5yr.csv", parse_dates=['date'])
    df = df.sort_values(['Name', 'date'])

    df['return'] = df.groupby('Name')['close'].transform(lambda x: ((x / x.shift(1)) - 1))

    volatility = df.groupby('Name')['return'].std() * np.sqrt(252)

    kde = gaussian_kde(volatility)

    kde_x = np.linspace(0, 1, 1000)
    kde_y = kde(kde_x)

    shape, loc, scale = lognorm.fit(volatility, floc=0)

    mean, std = lognorm_dist = lognorm(s=shape, loc=loc, scale=scale).stats(moments="mv")
    print("Mean:", mean)
    print("Stdev:", std)

    plt.plot(kde_x, kde_y, label="KDE")
    #plt.hist(volatility, bins=30, edgecolor='k')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    y = lognorm.pdf(x, shape, loc=loc, scale=scale)
    plt.plot(x, y, label="Log Normal Fit")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Number of Stocks")
    plt.title("Distribution of Stock Volatility (S&P 500)")
    plt.legend()
    plt.show()