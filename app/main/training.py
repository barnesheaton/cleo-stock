from tqdm import tqdm
import yfinance as yf
import tensorflow as tf
import modin.pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import math
import mplcursors
import os
import pickle
import requests
import sys
import warnings
import itertools

from app.main.database import Database

import analysis
import utils

from hmmlearn.hmm import GaussianHMM

# Saving a Model should be bound with predicted features

def getTickerOutlook(model, possible_outcomes, ticker="AAPL"):
    # update to read form our DB
    dataframe = Database().getTickerData(ticker)
    dataframe = yf.download(tickers=ticker, period="max", group_by="ticker")

    getPredictions(
        model,
        dataframe,
        possible_outcomes=possible_outcomes,
        prediction_period=prediction_period
    )


def runVerfication(
    model,
    possible_outcomes,
    lookback_period=40,
    prediction_period=4,
    start=3,
    end=7,
    tickers=False,
    startDate="2019-01-01",
    endDate="2021-01-01"
):
    ticker_list, ticker_string = utils.getTickerList(
        start=start, end=end, tickers=tickers)
    yf_dataframe = yf.download(
        tickers=ticker_string, start=startDate, end=endDate, group_by="ticker")

    for ticker in ticker_list:
        # update to read form our DB
        dataframe = yf_dataframe.dropna() if (
            len(ticker_list) == 1) else yf_dataframe[ticker].dropna()

        if dataframe.shape[0] <= (lookback_period + prediction_period):
            continue

        utils.printLine(f"{ticker}")
        length = dataframe.shape[0]
        start_index = 0 if lookback_period == 0 else length - \
            (lookback_period + prediction_period)

        predictions_df = getPredictions(
            model,
            dataframe.iloc[start_index: -prediction_period],
            possible_outcomes=possible_outcomes,
            prediction_period=prediction_period
        )
        verification_df = dataframe.iloc[start_index:]

        utils.printLine("Predictions")
        print(predictions_df.tail(prediction_period + 5))
        print(verification_df.tail(prediction_period + 5))

        plotPredictedCloses(predictions_df, verification_df, ticker)


def getPredictions(model, dataframe, possible_outcomes, prediction_period=10):
    p_dataframe = dataframe
    for index in range(0, prediction_period):
        data = getFeatures(p_dataframe)

        high_price = p_dataframe.iloc[-1]['High']
        # low_price = p_dataframe.iloc[-1]['Low']
        open_price = p_dataframe.iloc[-1]['Open']
        close_price = p_dataframe.iloc[-1]['Close']

        # 20% chance of flipping the sign of the prediciton
        # mutiplier = -1 if np.random.pareto(1) > 1.2 else 1
        mutiplier = 1
        bands = analysis.getBollingerBandWidths(
            p_dataframe['Close'].to_numpy())
        delta_open, delta_close, _ = getPredictedFeatures(
            model,
            data,
            possible_outcomes=possible_outcomes
        )

        # hit_upper_band = close_price >= (bands['upper'][-1] * 0.96) or open_price >= (bands['upper'][-1] * 0.96)
        # hit_lower_band = close_price <= (bands['lower'][-1] * 1.04) or open_price <= (bands['lower'][-1] * 1.04)
        hit_upper_band = close_price >= (bands['upper'][-1] * 0.96)
        hit_lower_band = close_price <= (bands['lower'][-1] * 1.04)

        if hit_upper_band and delta_close >= 0:
            print("Hit UPPER band")
            mutiplier = -0.5
        if hit_lower_band and delta_close <= 0:
            print("Hit LOWER band")
            mutiplier = -0.5

        predicted_open = close_price * (1 + (delta_open))
        # predicted_d_high = predicted_open * (1 + frac_high)
        predicted_close = predicted_open * (1 + (delta_close * mutiplier))

        p_dataframe = pd.concat([p_dataframe, pd.DataFrame({
            'Open': [predicted_open],
            'High': [0],
            'Low': [0],
            'Close': [predicted_close],
            'Adj Close': [0],
            'Volume': [0],
        })])

    return p_dataframe


def getPredictedFeatures(model, data, possible_outcomes):
    outcome_score = []
    for possible_outcome in tqdm(possible_outcomes):
        total_data = np.row_stack((data, possible_outcome))
        outcome_score.append(model.score(total_data))

    return possible_outcomes[np.argmax(outcome_score)]


def getFeatures(dataframe):
    open_price = np.array(dataframe['Open'])
    close_price = np.array(dataframe['Close'])

    delta_open = (open_price[1:] - close_price[0:-1]) / close_price[0:-1]
    frac_change = (close_price - open_price) / open_price
    rsis = analysis.getReltiveStrengthIndexes(close_price)
    rsis_length = len(rsis)

    return np.column_stack((
        delta_open[-rsis_length:],
        frac_change[-rsis_length:],
        np.array(rsis)
    ))


def getPossibleOutcomes(n_steps_delta_open=20, n_steps_delta_close=20, n_steps_rsis=80):
    delta_open_range = np.linspace(-0.2, 0.2, n_steps_delta_open)
    delta_close_range = np.linspace(-0.2, 0.2, n_steps_delta_close)
    rsis_range = np.linspace(1, 100, n_steps_rsis)

    return np.array(list(itertools.product(delta_open_range, delta_close_range, rsis_range)))


def plotPredictedCloses(predictions, verifications, ticker):
    fig = plt.figure()
    axes = fig.add_subplot(111)

    axes.plot(range(predictions.shape[0]),
              predictions['Close'], '--', color='red')
    axes.plot(
        range(verifications.shape[0]), verifications['Close'], color='black', alpha=0.5)