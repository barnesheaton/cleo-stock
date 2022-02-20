from flask import current_app
from app import db
from app.models import Task
import datetime

class Session():
    def create_task(self, name):
        task = Task(name=name, date=datetime.datetime.today())
        db.session.add(task)
        db.session.commit()
        return task

    def launch_task(self, task, *args, **kwargs):
        rq_job = current_app.task_queue.enqueue('app.tasks.' + task.name, *args, **kwargs)
        task.job_id = rq_job.get_id()
        db.session.commit()

    def create_and_launch_task(self, name, *args, **kwargs):
        rq_job = current_app.task_queue.enqueue('app.tasks.' + name, *args, **kwargs)
        task = Task(name=name, job_id=rq_job.get_id(), date=datetime.datetime.today())
        db.session.add(task)
        db.session.commit()
        return task

    def get_all_tasks(self):
        return Task.query.order_by(Task.date).all()

    def get_tasks_in_progress(self):
        return Task.query.filter_by(complete=False).all()

    def get_tasks_completed(self):
        return Task.query.filter_by(complete=True).all()

    def get_task(self, name):
        return Task.query.filter_by(name=name).all()
