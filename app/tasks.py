import pickle
import datetime
from random import sample
from rq import get_current_job

from app.main.database import Database
from app.main.training import trainModel
from app.main.training import simulate

from app.models import StockModel
from app import db

from wtforms.validators import ValidationError

def updateTickerTablesTask(*args, **kwargs):
    Database().updateTickerTables(*args, **kwargs)
    
def simulateTask(*args, **kwargs):
    simulate(*args, **kwargs)

def trainModelTask(modelName, tickerString=None, samplePercent=None, modelDescription=None):
    if samplePercent:
        tickers = Database().getTickerTablesList(samplePercent=samplePercent)

    elif tickerString:
        tickers = Database().getTickerTablesList(tickerString=tickerString)

    if len(tickers) < 1:
        raise ValidationError("Ticker selection or sampling method must have at least 1 ticker mathced in the database") 

    dataframe = Database().getTrainingData(tickers)
    model = trainModel(dataframe)
    pickleString = pickle.dumps(model)

    newModel = StockModel(name=modelName, description=modelDescription, tickers=tickers, pickle=pickleString, date=datetime.datetime.today())
    db.session.add(newModel)   
    db.session.commit()
