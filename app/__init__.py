from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
import redis
import rq
import os

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

from app import routes, models

def create_app(config_class=Config):
    db.init_app(app)
    app.redis = redis.StrictRedis(app.config['REDIS_URL'], app.config['REDIS_PORT'], 0)
    app.task_queue = rq.Queue('cleo-tasks', connection=app.redis)
    return app
