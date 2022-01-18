from tqdm import tqdm
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime
import os
import pickle
import itertools

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
    simulation = Simulation(model_id=model_id, date=datetime.date.today(), start_date=start_date, end_date=end_date, starting_capital=principal, complete=False)
    db.session.add(simulation)   
    db.session.commit()
    currentDay = start_date

    possible_outcomes = getPossibleOutcomes()
    stockModel = StockModel.query.get(model_id)
    loadedModel = pickle.loads(stockModel.pickle)

    databaseTickers = Database().getTickerTablesList()
    modelTickers = stockModel.tickers
    simulationTickers = utils.xor(databaseTickers, modelTickers)

    # -------- Main Simulation Loop --------
    utils.printLine("Accuracy Verification")
    print('Starting Principal :: ', principal)
    print('Prediction Period (days) :: ', prediction_period)
    print('Lookback Period (days) :: ', lookback_period)
    print('Simulations Tickers', simulationTickers)

    openAccuracy = np.array([])
    closeAccuracy = np.array([])

    while currentDay <= end_date:
        utils.printLine(f"Day ({currentDay})")
        nextDay = currentDay + timedelta(days=1)
        for ticker in simulationTickers:
            print(f"[=== Ticker {ticker} ===]")
            lookbackDF = Database().getTickerDataToDate(ticker, currentDay, lookback_period)
            verificationDF = Database().getTickerDataAfterDate(ticker, nextDay, prediction_period)
            print('lookback Dataframe\n', lookbackDF.tail(3))
            print('verification Dataframe\n', verificationDF.head(3))
            if lookbackDF.shape[0] < lookback_period:
                print('Not enough rows for T.A., skipping')
                continue

            predictionDF = getPredictions(model=loadedModel, dataframe=lookbackDF, possible_outcomes=possible_outcomes, prediction_period=prediction_period)
            maxDiff = getMaxDiffInPrediction(lookbackDF.iloc[-1]['close'], predictionDF, predictionPeriod=prediction_period)
            print('prediction Dataframe\n', predictionDF.tail(3))
            print('Max Price Delta :: ', maxDiff)

            predictedOpens = np.array(predictionDF.iloc[-prediction_period:]['open'])
            verfiedOpens = verificationDF['open'].to_numpy()
            openPercentages = np.abs((predictedOpens - verfiedOpens) / verfiedOpens)
            openAccuracy = np.concatenate([openAccuracy, openPercentages], axis=None)

            predictedCloses = np.array(predictionDF.iloc[-prediction_period:]['close'])
            verfiedCloses = verificationDF['close'].to_numpy()
            closePercentages = np.abs((predictedCloses - verfiedCloses) / verfiedCloses)
            closeAccuracy = np.concatenate([closeAccuracy, closePercentages], axis=None)

        currentDay = nextDay

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

# TODO add absulte value to deltas to account for dips as well
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

def getPredictions(model, dataframe, possible_outcomes, prediction_period=10):
    p_dataframe = dataframe
    for index in range(0, prediction_period):
        data = getFeatures(p_dataframe)

        high_price = p_dataframe.iloc[-1]['high']
        # low_price = p_dataframe.iloc[-1]['Low']
        open_price = p_dataframe.iloc[-1]['open']
        close_price = p_dataframe.iloc[-1]['close']

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

        if hit_upper_band and delta_close >= 0:
            # print("Hit UPPER band")
            mutiplier = -0.5
        if hit_lower_band and delta_close <= 0:
            # print("Hit LOWER band")
            mutiplier = -0.5

        predicted_open = close_price * (1 + (delta_open))
        # predicted_d_high = predicted_open * (1 + frac_high)
        predicted_close = predicted_open * (1 + (delta_close * mutiplier))

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
