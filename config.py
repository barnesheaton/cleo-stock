import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    FLASK_APP = os.environ.get('FLASK_APP') or 'cleo.py'
    FLASK_ENV = os.environ.get('FLASK_ENV') or 'development'
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://'
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secret'

    BUCKETEER_AWS_ACCESS_KEY_ID = os.environ.get('BUCKETEER_AWS_ACCESS_KEY_ID')
    BUCKETEER_AWS_REGION = os.environ.get('BUCKETEER_AWS_REGION')
    BUCKETEER_AWS_SECRET_ACCESS_KEY = os.environ.get('BUCKETEER_AWS_SECRET_ACCESS_KEY')
    BUCKETEER_BUCKET_NAME = os.environ.get('BUCKETEER_BUCKET_NAME')

    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql:///app' # or 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
