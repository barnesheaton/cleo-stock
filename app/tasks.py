import time
import boto3
import os
import pickle
from rq import get_current_job
from app.main.database import Database
from app.main.training import trainModel
from app.main.training import simulate
from app import app

def updateTickerTablesTask(*args, **kwargs):
    Database().updateTickerTables(*args, **kwargs)
    
def simulateTask(*args, **kwargs):
    simulate(*args, **kwargs)

def trainModelTask(*args, **kwargs):
    database = Database()
    dataframe = database.getTrainingData()
    model = trainModel(dataframe)

    # TODO use current datetime for filename
    filename = 'test_file_one'
    outfile = open(os.path.join(app.config['MODELS_DR'], filename), 'wb')
    file = os.path.join(app.config['MODELS_DR'], filename)
    pickle.dump(model, outfile)

    s3_client = boto3.resource(
        's3',
        aws_access_key_id=app.config['BUCKETEER_AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=app.config['BUCKETEER_AWS_SECRET_ACCESS_KEY'],
    )

    for bucket in s3_client.buckets.all():
        print(bucket.name)

    response = s3_client.upload_file(file, app.config['BUCKETEER_BUCKET_NAME'])
    outfile.close()

    return response

