import time
import pickle
import os
from rq import get_current_job
from app.main.database import Database
from app.main.training import trainModel
from app import app

def updateTickerTablesTask(*args, **kwargs):
    database = Database()
    database.updateTickerTables(*args, **kwargs)

def trainModelTask(*args, **kwargs):
    model = trainModel()
    model_count = len(os.listdir(app.config['MODELS_DR']))
    filename = f"model_{model_count + 1}"
    outfile = open(os.path.join(app.config['MODELS_DR'], filename), 'wb')
    pickle.dump(model, outfile)
    outfile.close()
