import numpy as np

def getBollingerBandWidths(prices, period=14):
    if len(prices) < period + 2:
        return {}

    upper_band = []
    lower_band = []
    widths = []
    band_length = len(prices) - period + 1
    for index in range(0, band_length):
        sma = np.mean(prices[index: index + period])
        sd = np.std(prices[index: index + period])
        upper = sma + (sd * 2)
        lower = sma - (sd * 2)
        upper_band.append((sma + (sd * 2)))
        lower_band.append((sma - (sd * 2)))
        widths.append(upper - lower)

    # return widths
    return {
        'upper': upper_band,
        'lower': lower_band,
        # 'widths': widths
    }


def getReltiveStrengthIndexes(prices, period=14):
    rsi_length = len(prices) - period
    if len(prices) < period + 2:
        return []

    def rs(g, l): return 100 if l == 0 else g / l
    def rsi(r): return 100 - (100 / (1 + r))

    gains = []
    losses = []
    # shift and subtract close prices from eachother
    changes = prices[1:] - prices[0:-1]
    for c in changes:
        gains.append(c if c > 0 else 0)
        losses.append(np.abs(c if c < 0 else 0))

    # First avg gain, avg loss, and RSI
    abv_gain_0 = np.mean(gains[:period])
    abv_loss_0 = np.mean(losses[:period])
    avg_gains = [abv_gain_0]
    avg_losses = [abv_loss_0]
    rsis = [rsi(rs(abv_gain_0, abv_loss_0))]

    for index in range(0, rsi_length - 1):
        new_avg_gain = (
            (avg_gains[index] * (period - 1)) + gains[period + index]) / period
        new_avg_loss = (
            (avg_losses[index] * (period - 1)) + losses[period + index]) / period
        new_rsi = rsi(rs(new_avg_gain, new_avg_loss))

        avg_losses.append(new_avg_loss)
        avg_gains.append(new_avg_gain)
        rsis.append(new_rsi)

    return rsis


def getExponentialMovingAverages(prices, period=12, smoothing=1):
    emas_length = len(prices) - period + 1
    if len(prices) < period + 2:
        return []

    multiplier = smoothing / (1 + period)
    def ema(price, old_ema): return (
        price * multiplier) + (old_ema * (1 - multiplier))

    emas = [np.mean(prices[:period])]
    for index in range(0, emas_length - 1):
        emas.append(ema(prices[period + index], emas[index]))

    return emas


def getStochasticOscillator(close, highs, lows, period=14):
    high = np.max(highs[-period:])
    low = np.min(lows[-period:])

    if high - low == 0:
        return 50

    return ((close - low) / (high - low)) * 100
