import matplotlib.pyplot as plt


def plot_histogram(df):
    final_prices = df.iloc[-1]

    plt.figure(figsize=(10, 6))
    plt.hist(final_prices, bins=100, edgecolor='black', alpha=0.7)
    plt.axvline(final_prices.mean(), linestyle='--', linewidth=2,
                label=f'Mean: {final_prices.mean():.2f}')
    plt.axvline(final_prices.median(), linestyle='--', linewidth=2,
                label=f'Median: {final_prices.median():.2f}')
    plt.legend()
    plt.title("Distribution of Final Stock Prices")
    plt.show()


def plot_paths(df):
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], alpha=0.3, linewidth=0.5)
    plt.title("Simulated Price Paths")
    plt.show()