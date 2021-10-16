from flask import current_app
from app import db
from app.models import Task

class Session():
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
