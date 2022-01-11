from flask import current_app
from app import db
import redis
import rq

class Task(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(128), index=True)
    description = db.Column(db.String(128))
    complete = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"Task - ${self.name} Complete => [${self.complete}]"

    def get_rq_job(self):
        try:
            rq_job = rq.job.Job.fetch(self.id, connection=current_app.redis)
        except (redis.exceptions.RedisError, rq.exceptions.NoSuchJobError):
            return None
        return rq_job

    def get_progress(self):
        job = self.get_rq_job()
        return job.meta.get('progress', 0) if job is not None else 100

class StockModel(db.Model):
    id = db.Column('id', db.Integer(), primary_key=True, autoincrement=True)
    tickers = db.Column(db.String())
    features = db.Column(db.Integer())
    pickle = db.Column(db.PickleType())

# class Simulation(db.Model):
#     id = db.Column(db.String(36), primary_key=True)
#     starting_capital = db.Column(db.Float())
#     ending_capital = db.Column(db.Float())
#     simulation_period = db.Column(db.Integer())
#     # settings = db.Column(db.Json())

class Bar(db.Model):
    date = db.Column(db.String(128), primary_key=True)
    open = db.Column(db.Float())
    high = db.Column(db.Float())
    low = db.Column(db.Float())
    close = db.Column(db.Float())
    adj_close = db.Column(db.Float())
    volume = db.Column(db.Float())

    def __repr__(self):
        return f"Bar: [{self.ticker}] || Date: [{self.date}] || Close: [{self.close}]"

# class PredictedStock(db.Model):
#     prediction_id = db.Column(db.String(36), primary_key=True)
#     ticker = db.Column(db.String(128))
#     date = db.Column(db.String(128))
#     open = db.Column(db.Float())
#     high = db.Column(db.Float())
#     low = db.Column(db.Float())
#     close = db.Column(db.Float())
#     adj_close = db.Column(db.Float())
#     volume = db.Column(db.Float())

# class Transaction(db.Model):
#     id = db.Column(db.String(36), primary_key=True)
#     ticker = db.Column(db.String(128))
