import pickle
import datetime
from random import sample
from rq import get_current_job

from app.main.database import Database
from app.main.training import trainModel
from app.main.training import simulate, plotVerificaitonForTicker

from app.models import StockModel
from app import db

from wtforms.validators import ValidationError

def updateTickerTablesTask(*args, **kwargs):
    Database().updateTickerTables(*args, **kwargs)
    
def simulateTask(*args, **kwargs):
    simulate(*args, **kwargs)

def plotTask(ticker, model_id, prediction_period, lookback_period):
    plotVerificaitonForTicker(ticker, model_id, prediction_period, lookback_period)

def trainModelTask(model_name, observation_period, tickerString=None, sample_percent=None, model_description=None):
    if sample_percent:
        tickers = Database().getTickerTablesList(sample_percent=sample_percent)

    elif tickerString:
        tickers = Database().getTickerTablesList(tickerString=tickerString)

    if len(tickers) < 1:
        raise ValidationError("Ticker selection or sampling method must have at least 1 ticker existing in the database") 

    dataframe = Database().getTrainingData(tickers)
    model = trainModel(dataframe, observation_period)
    pickle_string = pickle.dumps(model)

    newModel = StockModel(
        name=model_name,
        description=model_description,
        observation_period=observation_period,
        tickers=tickers,
        pickle=pickle_string,
        date=datetime.datetime.today()
    )
    db.session.add(newModel)   
    db.session.commit()
