from unittest import result
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import math
import pickle
import itertools
import time

from app import app, db

import app.main.utils as utils
import app.main.analysis as analysis
from app.main.database import Database
from app.models import StockModel, Simulation
from numpy import vectorize

from hmmlearn.hmm import GaussianHMM

def trainModel(dataframe):
    X = getFeatures(dataframe)
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=20)
    model.fit(X)
    return model

def makeTransaction(positions, ticker, price, shares, date, capital, selling=True):
    print(f"{'SOLD' if selling else 'BOUGHT'} [{ticker}] for {price} x({shares})")
    positions[ticker] = { 'price': price, 'shares': shares, 'date': date, 'sold': selling }
    return (1 if selling else -1) * price * shares

# Saving a Model should be bound with predicted features
def simulate(
    model_id=3,
    lookback_period=0,
    prediction_period=14,
    start_date="2019-01-01",
    end_date="2019-12-31",
    principal=10000,
    diversification=5
):
    simulation = Simulation(model_id=model_id, date=datetime.today(), start_date=start_date, end_date=end_date, starting_capital=principal, complete=False)
    db.session.add(simulation)   
    db.session.commit()
    current_day = start_date

    possible_outcomes = getPossibleOutcomes()
    stockModel = StockModel.query.get(model_id)
    loadedModel = pickle.loads(stockModel.pickle)

    databaseTickers = Database().getTickerTablesList()
    modelTickers = stockModel.tickers
    simulationTickers = ['aapl'] or utils.xor(databaseTickers, modelTickers)

    # -------- Main Simulation Loop --------
    utils.printLine("Accuracy Verification")
    print('Starting Principal :: ', principal)
    print('Prediction Period (days) :: ', prediction_period)
    print('Lookback Period (days) :: ', lookback_period)
    print('Simulation Tickers', simulationTickers)

    openAccuracy = np.array([])
    closeAccuracy = np.array([])

    capital = principal
    positions = {}
    max_buys_per_day = 5
    prospects_on_day = []
    while current_day <= end_date:
        utils.printLine(f"Day ({current_day})")
        next_day = current_day + timedelta(days=1)

        # --- Buying Stage ---
        for ticker in prospects_on_day:
            ticker_data_on_date = Database().getTickerDataToDate(ticker, current_day, 1)
            open_price = ticker_data_on_date['open'][0]
            # Can buy at least one share of this prospect
            volume = math.floor((capital / len(prospects_on_day)) / open_price)
            if ticker not in positions and volume > 0:
                capital += makeTransaction(positions, ticker, open_price, volume, current_day, capital, selling=False)

        # --- Selling Stage ---
        portfolio_worth = 0
        for ticker in positions:
            if positions[ticker]['sold']:
                continue

            ticker_data_on_date = Database().getTickerDataToDate(ticker, current_day, 1)
            high_price = ticker_data_on_date['high'][0]
            low_price = ticker_data_on_date['low'][0]
            close_price = ticker_data_on_date['close'][0]
            price = positions[ticker]['price']
            shares = positions[ticker]['shares']
            date = positions[ticker]['date']
            # TODO make configurable
            stop_loss = 0.95 * price
            target_price = 1.25 * price
            is_last_day = current_day == end_date

            if low_price <= stop_loss and date != current_day:
                capital += makeTransaction(positions, ticker, stop_loss, shares, current_day, capital)
            elif high_price >= target_price:
                capital += makeTransaction(positions, ticker, target_price, shares, current_day, capital)
            elif close_price >= target_price or is_last_day:
                capital += makeTransaction(positions, ticker, close_price, shares, current_day, capital)
            else:
                portfolio_worth += close_price * shares

        # --- Prediction Stage ---
        prospects_on_day = []
        for ticker in simulationTickers:
            result = getTickerAccuracyAndPredictionMetrics(loadedModel, ticker, current_day, lookback_period, prediction_period, possible_outcomes)
            if not result:
                continue
           
            openAccuracy = np.concatenate([openAccuracy, result[0]], axis=None)
            closeAccuracy = np.concatenate([closeAccuracy, result[1]], axis=None)
            # Ticker gained 10% at some point in prediction period. TODO -> make configurable
            if result[2] >= 0.1:
                prospects_on_day.append(ticker)

        if (len(prospects_on_day) > max_buys_per_day):
            prospects_on_day = np.random.choice(prospects_on_day, max_buys_per_day)

        utils.printLine('EOD Stats')
        print("Capital", capital)
        print("Portfolio", portfolio_worth)
        print("Net Worth", capital + portfolio_worth)
        print("Positions", list(positions.keys()))
        current_day = next_day

    closeAccuracy = np.mean(closeAccuracy)
    openAccuracy = np.mean(openAccuracy)
    totalAccuracy = np.mean([closeAccuracy, openAccuracy])

    simulation.complete = True
    simulation.close_accuracy = closeAccuracy
    simulation.open_accuracy = openAccuracy
    simulation.total_accuracy = totalAccuracy
    db.session.commit()

    utils.printLine("Results")
    print('Close Accuracy :: ', closeAccuracy)
    print('Open Accuracy :: ', openAccuracy)
    print('Total Accuracy :: ', totalAccuracy)

    return

def getTickerAccuracyAndPredictionMetrics(model, ticker, date, lookback_period, prediction_period, possible_outcomes):
    tic = time.perf_counter()
    print(f"[=== Ticker ({ticker}) ===]")
    lookback_dataframe = Database().getTickerDataToDate(ticker, date, lookback_period)
    verification_dataframe = Database().getTickerDataAfterDate(ticker, date + timedelta(days=1), prediction_period)
    if lookback_dataframe.shape[0] < 21:
        print("Not enough data in Lookback Dataframe, skipping")
        return None
    predictionDF = getPredictions(model=model, dataframe=lookback_dataframe, possible_outcomes=possible_outcomes, prediction_period=prediction_period)
    print()
    maxDiff = getMaxDiffInPrediction(lookback_dataframe.iloc[-1]['close'], predictionDF, predictionPeriod=prediction_period)
    # --- Accuracy Calculations ---
    verfied_opens = verification_dataframe['open'].to_numpy()
    verfied_closes = verification_dataframe['close'].to_numpy()
    predicted_opens = np.array(predictionDF.iloc[-prediction_period:]['open'])
    predicted_closes = np.array(predictionDF.iloc[-prediction_period:]['close'])
    openPercentages = np.abs((predicted_opens - verfied_opens) / verfied_opens)
    closePercentages = np.abs((predicted_closes - verfied_closes) / verfied_closes)

    toc = time.perf_counter()
    print("getTickerAccuracyAndPredictionMetrics Performance :: ", toc - tic)

    return openPercentages, closePercentages, maxDiff

# TODO add absolute value to deltas to account for dips as well
def getMaxDiffInPrediction(price, dataframe, predictionPeriod=14):
    closeDeltas = (np.array(dataframe.iloc[-predictionPeriod:]['close']) - price) / price
    openDeltas = (np.array(dataframe.iloc[-predictionPeriod:]['open']) - price) / price
    return np.nanmax(np.concatenate([closeDeltas, openDeltas]))

# TODO may have to be based on user input eventually when model features can be chosen
def getPossibleOutcomes(n_steps_delta_open=20, n_steps_delta_close=20, n_steps_rsis=80):
    delta_open_range = np.linspace(-0.2, 0.2, n_steps_delta_open)
    delta_close_range = np.linspace(-0.2, 0.2, n_steps_delta_close)
    rsis_range = np.linspace(1, 100, n_steps_rsis)

    return np.array(list(itertools.product(delta_open_range, delta_close_range, rsis_range)))
    
def getTickerOutlook(possible_outcomes, model_id=3, ticker="aapl", prediction_period=14):
    dataframe = Database().getTickerData(ticker)
    stockModel = StockModel.query.get(model_id)
    loadedModel = pickle.loads(stockModel.pickle)

    predicitons = getPredictions(
        loadedModel,
        dataframe,
        possible_outcomes=possible_outcomes,
        prediction_period=prediction_period
    )

def getPredictionsOptimized(model, dataframe, possible_outcomes, prediction_period=14):
    results = np.array([])
    print("START results", results)
    for _ in itertools.repeat(None, prediction_period):
        data = getFeatures(p_dataframe)
        # Reduce to one iloc
        close_price = p_dataframe.iloc[-1]['close']
        # high_price = p_dataframe.iloc[-1]['high']
        # low_price = p_dataframe.iloc[-1]['Low']
        # open_price = p_dataframe.iloc[-1]['open']

        # TODO make this configurable in Model Settings, can be a Boolean on or off for Bollinger Bands
        # 20% chance of flipping the sign of the prediciton
        # mutiplier = -1 if np.random.pareto(1) > 1.2 else 1
        mutiplier = 1
        bands = analysis.getBollingerBandWidths(
            p_dataframe['close'].to_numpy())
        delta_open, delta_close, _ = getPredictedFeatures(
            model,
            data,
            possible_outcomes=possible_outcomes
        )

        # hit_upper_band = close_price >= (bands['upper'][-1] * 0.96) or open_price >= (bands['upper'][-1] * 0.96)
        # hit_lower_band = close_price <= (bands['lower'][-1] * 1.04) or open_price <= (bands['lower'][-1] * 1.04)
        hit_upper_band = close_price >= (bands['upper'][-1] * 0.96)
        hit_lower_band = close_price <= (bands['lower'][-1] * 1.04)
        if hit_upper_band and delta_close >= 0: mutiplier = -0.5
        if hit_lower_band and delta_close <= 0: mutiplier = -0.5

        predicted_open = close_price * (1 + (delta_open))
        predicted_close = predicted_open * (1 + (delta_close * mutiplier))
        # predicted_d_high = predicted_open * (1 + frac_high)

        # Use numpy here and concat array's instead of DF to improve performance?
        p_dataframe = pd.concat([p_dataframe, pd.DataFrame({
            'date': ['date'],
            'open': [predicted_open],
            'high': [0],
            'low': [0],
            'close': [predicted_close],
            'adj_close': [0],
            'volume': [0],
        })])

    return p_dataframe

def getPredictions(model, dataframe, possible_outcomes, prediction_period=10):
    p_dataframe = dataframe
    results = np.array([])
    for index in range(0, prediction_period):
        data = getFeatures(p_dataframe)
        close_price = p_dataframe.iloc[-1]['close']
        # high_price = p_dataframe.iloc[-1]['high']
        # low_price = p_dataframe.iloc[-1]['Low']
        # open_price = p_dataframe.iloc[-1]['open']

        # TODO make this configurable in Model Settings, can be a Boolean on or off for Bollinger Bands
        # 20% chance of flipping the sign of the prediciton
        # mutiplier = -1 if np.random.pareto(1) > 1.2 else 1
        mutiplier = 1
        bands = analysis.getBollingerBandWidths(
            p_dataframe['close'].to_numpy())
        delta_open, delta_close, _ = getPredictedFeatures(
            model,
            data,
            possible_outcomes=possible_outcomes
        )

        # hit_upper_band = close_price >= (bands['upper'][-1] * 0.96) or open_price >= (bands['upper'][-1] * 0.96)
        # hit_lower_band = close_price <= (bands['lower'][-1] * 1.04) or open_price <= (bands['lower'][-1] * 1.04)
        hit_upper_band = close_price >= (bands['upper'][-1] * 0.96)
        hit_lower_band = close_price <= (bands['lower'][-1] * 1.04)
        if hit_upper_band and delta_close >= 0: mutiplier = -0.5
        if hit_lower_band and delta_close <= 0: mutiplier = -0.5

        predicted_open = close_price * (1 + (delta_open))
        predicted_close = predicted_open * (1 + (delta_close * mutiplier))
        # predicted_d_high = predicted_open * (1 + frac_high)

        # Use numpy here and concat array's instead of DF to improve performance?
        p_dataframe = pd.concat([p_dataframe, pd.DataFrame({
            'date': ['date'],
            'open': [predicted_open],
            'high': [0],
            'low': [0],
            'close': [predicted_close],
            'adj_close': [0],
            'volume': [0],
        })])

    return p_dataframe

def getPredictedFeatures(model, data, possible_outcomes):
    outcome_score = []
    isProduction = app.config['FLASK_ENV'] == 'production'
    for possible_outcome in tqdm(possible_outcomes, disable=isProduction):
        total_data = np.row_stack((data, possible_outcome))
        outcome_score.append(model.score(total_data))

    return possible_outcomes[np.argmax(outcome_score)]

def getFeatures(dataframe):
    open_price = np.array(dataframe['open'])
    close_price = np.array(dataframe['close'])

    def fc(c, o): return 100 if o == 0 else (c - o) / o

    vfunc = vectorize(fc)

    delta_open = (open_price[1:] - close_price[0:-1]) / close_price[0:-1]
    frac_change = vfunc(close_price, open_price)
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
              predictions['close'], '--', color='red')
    axes.plot(
        range(verifications.shape[0]), verifications['close'], color='black', alpha=0.5)
