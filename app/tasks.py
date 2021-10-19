import time
import pickle
import os
from rq import get_current_job
from app.main.database import Database
from app.main.training import trainModel
from app import app

def example(seconds):
    job = get_current_job()
    print('Starting task')
    for i in range(seconds):
        job.meta['progress'] = 100.0 * i / seconds
        job.save_meta()
        print(i)
        time.sleep(1)
    job.meta['progress'] = 100
    job.save_meta()
    print('Task completed')

def example(seconds):
    job = get_current_job()
    print('Starting task')
    for i in range(seconds):
        job.meta['progress'] = 100.0 * i / seconds
        job.save_meta()
        print(i)
        time.sleep(1)
    job.meta['progress'] = 100
    job.save_meta()
    print('Task completed')

def updateTickerTablesTask(*args, **kwargs):
    database = Database()
    database.updateTickerTables(*args, **kwargs)

def trainModelTask(*args, **kwargs):
    model = trainModel()
    filename = 'test_file_one'
    outfile = open(os.path.join(app.config['MODELS_DR'], filename), 'wb')
    pickle.dump(model, outfile)
    outfile.close()
