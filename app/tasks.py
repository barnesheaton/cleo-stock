import pickle
import datetime
from random import sample

from rq import get_current_job
from app.main.database import Database
from app.main.training import trainModel
from app.main.training import simulate
from app.main.training import trainModel

from app.models import StockModel


from app import db

def updateTickerTablesTask(*args, **kwargs):
    Database().updateTickerTables(*args, **kwargs)
    
def simulateTask(*args, **kwargs):
    simulate(*args, **kwargs)

def trainModelTask(modelName, tickers=None, samplePercent=50, modelDescription=None, *args, **kwargs):
    # TODO add validation

    tickers = Database().getTickerTablesList(samplePercent=samplePercent)
    dataframe = Database().getTrainingData(tickers)
    model = trainModel(dataframe)
    pickleString = pickle.dumps(model)


    newModel = StockModel(name=modelName, description=modelDescription, tickers=tickers, pickle=pickleString, date=datetime.date.today())
    db.session.add(newModel)   
    db.session.commit()
