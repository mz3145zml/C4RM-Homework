"""C4RM Class 06 homework: WhoAmI, returns, VaR, and expected shortfall.

Dependencies: numpy and matplotlib. Install yfinance only when downloading the
optional example stock data.
"""

from __future__ import annotations

import numpy as np


def WhoAmI():
    """Return the course identifier specified in the homework template."""
    return "zn2106"


def get_stock_data(symbol: str = "GS"):
    """Download a symbol's daily closing prices with yfinance."""
    import yfinance as yf

    data = yf.download(symbol, progress=False, auto_adjust=False)
    if data.empty or "Close" not in data:
        raise ValueError(f"No closing-price data was returned for {symbol!r}.")
    prices = data["Close"]
    # yfinance may return a one-column DataFrame for a single ticker.
    if getattr(prices, "ndim", 1) == 2:
        prices = prices.iloc[:, 0]
    return prices


def get_returns(pricevec):
    """Calculate simple period returns from a sequence of prices.

    Return at t is price[t] / price[t-1] - 1.
    """
    prices = np.asarray(pricevec, dtype=float).reshape(-1)
    if prices.size < 2:
        raise ValueError("pricevec must contain at least two prices.")
    if not np.all(np.isfinite(prices)):
        raise ValueError("pricevec must contain only finite values.")
    if np.any(prices[:-1] == 0):
        raise ValueError("A prior price cannot be zero when calculating returns.")
    return prices[1:] / prices[:-1] - 1


def percent_var(r, alpha):
    """Return positively stated historical VaR and display a return histogram.

    ``alpha`` is the confidence level (for example, 0.95). The VaR threshold
    is the absolute value of the (1 - alpha) left-tail return percentile.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    returns = np.asarray(r, dtype=float).reshape(-1)
    if returns.size == 0 or not np.all(np.isfinite(returns)):
        raise ValueError("r must contain at least one finite return.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive).")

    if plt is not None:
        plt.hist(returns, bins=50, alpha=0.75)
        plt.title("Distribution of Returns")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("Install matplotlib to display the returns histogram.")

    left_tail_return = np.percentile(returns, (1 - alpha) * 100)
    return abs(float(left_tail_return))


def es(losses, alpha=None, VaR=None):
    """Calculate expected shortfall from positively stated losses.

    If ``VaR`` is omitted, it is calculated as the alpha percentile of
    the losses. ES is the mean of losses strictly greater than that threshold.
    """
    loss_values = np.asarray(losses, dtype=float).reshape(-1)
    if loss_values.size == 0 or not np.all(np.isfinite(loss_values)):
        raise ValueError("losses must contain at least one finite value.")

    if VaR is None:
        if alpha is None:
            raise ValueError("Provide alpha when VaR is not specified.")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive).")
        VaR = float(np.percentile(loss_values, 100 * alpha))
    elif not np.isfinite(VaR):
        raise ValueError("VaR must be a finite number.")

    tail_losses = loss_values[loss_values > VaR]
    # When all observations equal or fall below VaR, the empirical tail is empty.
    # Return VaR itself as the limiting empirical shortfall in that case.
    if tail_losses.size == 0:
        return float(VaR)
    return float(np.mean(tail_losses))


def _run_examples_and_checks():
    """Run the homework examples and simple reproducible checks."""
    assert WhoAmI() == "zn2106"

    sample_prices = np.array([100.0, 102.0, 101.0])
    sample_returns = get_returns(sample_prices)
    assert np.allclose(sample_returns, [0.02, 101.0 / 102.0 - 1])

    # VaR example from the assignment: a normal return sample with a 2-SD tail.
    rng = np.random.default_rng(6)
    simulated_returns = rng.normal(0.05, 0.03, 1_000_000)
    probability_2sd = 0.9772498680518208  # standard normal CDF at 2
    var_value = percent_var(simulated_returns, probability_2sd)
    assert round(var_value, 2) == 0.01
    print("VaR example passed:", round(var_value, 2) == 0.01)

    # ES examples from the assignment, using a seeded sample for reproducibility.
    uniform_losses = np.random.default_rng(6).uniform(0, 100, 100_000)
    es_alpha = es(losses=uniform_losses, alpha=0.8)
    es_var = es(losses=uniform_losses, VaR=80)
    assert round(es_alpha) == 90
    assert round(es_var) == 90
    print("ES with alpha passed:", round(es_alpha) == 90)
    print("ES with VaR passed:", round(es_var) == 90)

    # Print the requested summary statistics for downloaded GS closing prices.
    try:
        prices = get_stock_data("GS")
        returns = get_returns(prices)
        print("GS average return:", float(np.mean(returns)))
        print("GS maximum return:", float(np.max(returns)))
        print("GS minimum return:", float(np.min(returns)))
        print("GS standard deviation of returns:", float(np.std(returns)))
    except (ImportError, ModuleNotFoundError) as exc:
        print("Skipping live GS example; optional dependency unavailable:", exc)
    except Exception as exc:
        print("Skipping live GS example; stock-data download failed:", exc)


if __name__ == "__main__":
    _run_examples_and_checks()
