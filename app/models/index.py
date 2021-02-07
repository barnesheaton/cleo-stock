from flask import current_app
from app import db
import redis
import rq

class Stock(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    ticker=db.Column(db.String(128))
    date = db.Column(db.String(128))
    open = db.Column(db.Float())
    high = db.Column(db.Float())
    low = db.Column(db.Float())
    close = db.Column(db.Float())
    adj_close = db.Column(db.Float())
    volume = db.Column(db.Float())

    def __repr__(self):
        return f"Stock - ${self.ticker} - ${self.date}"

class PredictedStock(db.Model):
    prediction_id = db.Column(db.String(36), primary_key=True)
    ticker = db.Column(db.String(128))
    date = db.Column(db.String(128))
    open = db.Column(db.Float())
    high = db.Column(db.Float())
    low = db.Column(db.Float())
    close = db.Column(db.Float())
    adj_close = db.Column(db.Float())
    volume = db.Column(db.Float())

class Transaction(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    ticker = db.Column(db.String(128))
