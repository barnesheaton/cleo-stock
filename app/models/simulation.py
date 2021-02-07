from flask import current_app
from app import db
import redis
import rq

class Simulation(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    starting_capital = db.Column(db.Float())
    ending_capital = db.Column(db.Float())
    simulation_period = db.Column(db.Integer())
    # settings = db.Column(db.Json())
