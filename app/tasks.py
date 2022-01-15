import pickle
import datetime

from rq import get_current_job
from app.main.database import Database
from app.main.training import trainModel
from app.main.training import simulate
from app.models import StockModel
from app import db

def updateTickerTablesTask(*args, **kwargs):
    Database().updateTickerTables(*args, **kwargs)
    
def simulateTask(*args, **kwargs):
    simulate(*args, **kwargs)

def trainModelTask(*args, **kwargs):
    # TODO add validation
    tickers = Database().getTickerTablesList(samplePercent=50)
    dataframe = Database().getTrainingData(tickers)
    model = trainModel(dataframe)

    # TODO add current DateTime to StockModel
    pickleString = pickle.dumps(model)
    newModel = StockModel(tickers=tickers, pickle=pickleString, date=datetime.date.today())
    db.session.add(newModel)   
    db.session.commit()
