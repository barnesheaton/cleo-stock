import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    FLASK_APP = os.environ.get('FLASK_APP') or 'cleo.py'
    FLASK_ENV = os.environ.get('FLASK_ENV') or 'development'
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://'
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secret'

    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql:///app' # or 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
