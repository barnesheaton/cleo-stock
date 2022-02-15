from flask import current_app
from app import db
from app.models import Task, StockModel

class Session():
    def create_task(self, name):
        task = Task(name=name)
        db.session.add(task)
        db.session.commit()
        return task

    def launch_task(self, task, *args, **kwargs):
        current_app.task_queue.enqueue('app.tasks.' + task.name, *args, **kwargs)

    def create_and_launch_task(self, name, *args, **kwargs):
        print(name)
        current_app.task_queue.enqueue('app.tasks.' + name, *args, **kwargs)
        task = Task(name=name)
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

    def getAllModels(self):
        return StockModel.query.all()
