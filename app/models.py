from flask import current_app
from app import db
import redis
import rq

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def launch_task(self, name, *args, **kwargs):
        rq_job = current_app.task_queue.enqueue('app.tasks.' + name, *args, **kwargs)
        task = Task(id=rq_job.get_id(), name=name)
        db.session.add(task)
        db.session.commit()
        return task

    def get_all_tasks(self):
        return Task.query.all()

    def get_tasks_in_progress(self):
        return Task.query.filter_by(complete=False).all()
    
    def get_tasks_completed(self):
        return Task.query.filter_by(complete=True).all()

    def get_task(self, name):
        return Task.query.filter_by(name=name).first()


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
