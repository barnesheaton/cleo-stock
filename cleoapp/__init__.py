from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_bootstrap import Bootstrap

# import redis
from redis import Redis
import rq
import os

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bootstrap = Bootstrap(app)

from cleoapp import routes
from cleoapp.models import Task, Bar

def create_app(config_class=Config):
    db.init_app(app)
    app.redis = Redis.from_url(app.config['REDIS_URL'])
    # app.redis = redis.StrictRedis(app.config['REDIS_URL'], app.config['REDIS_PORT'], 0)
    app.task_queue = rq.Queue('cleo-tasks', connection=app.redis)
    # default_worker.work(burst=True)
    return app

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'Task': Task, 'Bar': Bar, }
