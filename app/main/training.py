import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime, timedelta
from hmmlearn.hmm import GaussianHMM
import matplotlib.dates as mdates
from pomegranate import HiddenMarkovModel, NormalDistribution

from app.main.database import Database
from app.main.utils import printLine, printData, xor
from app.models import StockModel, Simulation
from app import app, db

def trainModel(dataframe, observation_period=50):
    X = getObservationsFromTickerData(dataframe, observation_period)
    # X = np.expand_dims(X, axis = 0)
    print('Number of Observations :: ', X.shape[0])
    model = GaussianHMM(n_components=6, covariance_type="diag", n_iter=20)
    model.fit(X)
    return model

def trainPomegranteModel(tickers, observation_period):
    X = Database().getPomegranteTrainingData(tickers, observation_period)
    # print('Number of Observations :: ', X.shape[0])
    model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=6, X=X)
    model.fit(X)
    return model

def makeTransaction(positions, ticker, price, shares, date, capital, selling=True):
    print(f"{'SOLD' if selling else 'BOUGHT'} [{ticker}] for {price} x({shares})")
    positions[ticker] = { 'price': price, 'shares': shares, 'date': date, 'sold': selling }
    return (1 if selling else -1) * price * shares

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

    stockModel = StockModel.query.get(model_id)
    loaded_model = pickle.loads(stockModel.pickle)

    databaseTickers = Database().getTickerTablesList()
    modelTickers = stockModel.tickers
    simulation_tickers = xor(databaseTickers, modelTickers)

    # -------- Main Simulation Loop --------
    printLine("Accuracy Verification")
    print('Starting Principal :: ', principal)
    print('Prediction Period (days) :: ', prediction_period)
    print('Lookback Period (days) :: ', lookback_period)
    print('Simulation Tickers', simulation_tickers)

    closeAccuracy = np.array([])
    capital = principal
    positions = {}
    max_buys_per_day = diversification
    days_prospects = []
    while current_day <= end_date:
        printLine(f"Day ({current_day})")
        next_day = current_day + timedelta(days=1)

        # --- Buying Stage ---
        for ticker in days_prospects:
            ticker_data_on_date = Database().getTickerDataToDate(ticker, current_day, 1)
            open_price = ticker_data_on_date['open'][0]
            # Can buy at least one share of this prospect
            volume = math.floor((capital / len(days_prospects)) / open_price)
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
            stop_loss = 0.97 * price
            target_price = 1.1 * price
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
        days_prospects = []
        for ticker in simulation_tickers:
            result = getTickerAccuracyAndPredictionMetrics(
                stockModel,
                ticker,
                current_day,
                lookback_period,
                prediction_period,
                observation_period=stockModel.observation_period
            )
            if not result:
                continue
           
            closeAccuracy = np.concatenate([closeAccuracy, result[0]], axis=None)
            # Ticker gained 10% at some point in prediction period. TODO -> make configurable
            if result[1] >= 0.1:
                days_prospects.append(ticker)

        if (len(days_prospects) > max_buys_per_day):
            days_prospects = np.random.choice(days_prospects, max_buys_per_day)

        printLine('EOD Stats')
        printData('capital', capital)
        printData('portfolio_worth', portfolio_worth)
        printData('Net Worth', capital + portfolio_worth)
        print("Positions\n", list(positions.keys()))
        current_day = next_day

    closeAccuracy = np.mean(closeAccuracy)
    simulation.complete = True
    simulation.close_accuracy = closeAccuracy
    db.session.commit()

    printLine("Results")
    printData('closeAccuracy', closeAccuracy)

    return

def getTickerAccuracyAndPredictionMetrics(model, ticker, date, lookback_period, prediction_period, observation_period):
    print(f"[=== Ticker ({ticker}) ===]")
    lookback_dataframe = Database().getTickerDataToDate(ticker, date, lookback_period)
    verification_dataframe = Database().getTickerDataAfterDate(ticker, date + timedelta(days=1), prediction_period)
    if lookback_dataframe.shape[0] < observation_period + 1:
        print("Not enough data in Lookback Dataframe to make prediction, skipping")
        return None
    predictionDF = getPredictionsFromTickerData(model=model, dataframe=lookback_dataframe, prediction_period=prediction_period, observation_period=observation_period)
    maxDiff = getMaxDiffInPrediction(lookback_dataframe.iloc[-1]['close'], predictionDF, predictionPeriod=prediction_period)
    printLine('DataFrames')
    print(predictionDF.iloc[-prediction_period:], predictionDF.iloc[-prediction_period:].shape)
    print(verification_dataframe, verification_dataframe.shape)
    # --- Accuracy Calculations ---
    verfied_closes = verification_dataframe['close'].to_numpy()
    predicted_closes = np.array(predictionDF.iloc[-prediction_period:]['close'])
    closePercentages = np.abs((predicted_closes - verfied_closes) / verfied_closes)
    plotPredictedCloses(predictionDF.iloc[-prediction_period:], verification_dataframe)

    return closePercentages, maxDiff

# TODO add absolute value to deltas to account for dips as well
def getMaxDiffInPrediction(price, dataframe, predictionPeriod=14):
    closeDeltas = (np.array(dataframe.iloc[-predictionPeriod:]['close']) - price) / price
    return np.nanmax(closeDeltas)
    # openDeltas = (np.array(dataframe.iloc[-predictionPeriod:]['open']) - price) / price
    # return np.nanmax(np.concatenate([closeDeltas, openDeltas]))
    
def getTickerOutlook(model_id=3, ticker="aapl", prediction_period=14):
    dataframe = Database().getTickerData(ticker)
    stockModel = StockModel.query.get(model_id)
    # loaded_model = pickle.loads(stockModel.pickle)
    predicitons = getPredictionsFromTickerData(stockModel, dataframe, prediction_period=prediction_period)
    print('predicitons', predicitons)

def plotVerificaitonForTicker(tickers, task_id, model_id, prediction_period, lookback_period, limit=None):
    printLine('plotVerificaitonForTicker')
    plot_tickers = Database().getTickerTablesList(tickerString=tickers)
    stock_model = StockModel.query.get(model_id)
    # loaded_model = pickle.loads(stock_model.pickle)
    data_limit = int(limit) if limit else None

    print(plot_tickers)

    for ticker in plot_tickers:
        dataframe = Database().getTickerData(ticker, limit=data_limit)
        input_length = dataframe.shape[0]

        # Must be able to construct at least one sequence of observations
        # TODO move as much input verifcation to forms as possible
        if lookback_period < stock_model.observation_period + 1 or input_length < lookback_period:
            return

        # Make a prediction in increments of [prediction_period] along time axis of data
        increments = math.floor((input_length - lookback_period) / prediction_period)
        printLine(f'TICKER {ticker}')
        printData('increments', increments)
        printData('input_length', input_length)
        printData('lookback_period', lookback_period)
        printData('observation_period', stock_model.observation_period)
        printData('prediction_period', prediction_period)
        printData('task_id', task_id)
        for index in range(0, increments):
            # printLine('Predicting along increment')
            end_index = (index * prediction_period) + lookback_period
            input_data = dataframe.iloc[0 : end_index]

            prediction = getPredictionsFromTickerData(
                stock_model,
                dataframe=input_data,
                prediction_period=prediction_period,
                observation_period=stock_model.observation_period
            )
            # Set columns on predictions
            prediction['date'] = dataframe.iloc[0 : end_index + prediction_period ]['date'].to_numpy()
            prediction['ticker'] = ticker
            prediction['model_id'] = model_id
            prediction['task_id'] = task_id
            Database().savePredictions(prediction.iloc[-prediction_period:])

def getPredictionsFromTickerData(model, dataframe, prediction_period=10, observation_period=50):
    prediction_df = dataframe
    for index in range(0, prediction_period):
        observations = getObservationsFromTickerData(prediction_df, observation_period=observation_period)
        if index == 0:
            possible_outcomes = getPossibleOutcomesFromObservation(observations[-1], steps=4000, max_change_percent=0.01)
        else:
            possible_outcomes = getPossibleOutcomesFromObservation(observations[-1], steps=4000, max_change_percent=0.01)

        predicted_observation = getPredictedFeatures(model, observations, possible_outcomes)
        predicted_close = predicted_observation[-1]
        prediction_df = prediction_df.append(pd.Series({
            'date': 'date',
            'open': 0,
            'high': 0,
            'low': 0,
            'close': predicted_close,
            'adj_close': 0,
            'volume': 0
        }), ignore_index=True)

    return prediction_df

def getPossibleOutcomesFromObservation(observation, steps=1000, max_change_percent=0.03):
    # mean = np.mean(observation)
    start = observation[-1]
    possible_outcomes =[]
    frac_change_range = np.linspace(-1 * max_change_percent, max_change_percent, steps)
    isProduction = app.config['FLASK_ENV'] == 'production'
    for change in tqdm(frac_change_range, disable=isProduction):
        new_po = np.append(observation[1:], (start + (start * change)))
        possible_outcomes.append(new_po)

    return np.array(possible_outcomes)

# TODO ensure features match outcomes, with data and model
def getPredictedFeatures(model, observations, possible_outcomes):
    outcome_score = []
    isProduction = app.config['FLASK_ENV'] == 'production'
    loaded_model = pickle.loads(model.pickle)
    for possible_outcome in tqdm(possible_outcomes, disable=isProduction):
        total_data = np.row_stack((observations, possible_outcome))
        if model.model_type == 'pomegranate':
            # print('Scoring based on [Pomegranate]')
            outcome_score.append(loaded_model.log_probability(total_data))
        else:
            outcome_score.append(loaded_model.score(total_data))

    # print(f'Highest Prob. Outcome index: [{np.argmax(outcome_score)}]')
    return possible_outcomes[np.argmax(outcome_score)]

def getObservationsFromTickerData(dataframe, observation_period):
    observations = []
    observations_length = dataframe.shape[0] - observation_period
    # Input 1st observation into features array so numpy can properly concatenate
    isProduction = app.config['FLASK_ENV'] == 'production'
    for index in tqdm(range(0, observations_length), disable=isProduction):
        observations.append(dataframe.iloc[index : index + observation_period]['close'].to_numpy())

    return np.array(observations)

def plotPredictedCloses(predictions, verifications):
    fig = plt.figure()
    axes = fig.add_subplot(111)

    axes.plot(range(predictions.shape[0]), predictions['close'], '--', color='red')
    axes.plot(range(verifications.shape[0]), verifications['close'], color='black', alpha=0.5)
