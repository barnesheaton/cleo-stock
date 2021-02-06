from app import app
from flask import render_template, current_app
from app.forms import LoginForm, QueueForm
from app.models import User


@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', title='Home')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    return render_template('login.html', title='Sign In', form=form)


@app.route('/tasks', methods=['GET', 'POST'])
def tasks():
    form = QueueForm()
    if form.is_submitted():
        # rq_job = current_app.task_queue.enqueue('app.tasks.example', 23)
        # task = Task(id=rq_job.get_id(), name=name, description=description)
        u = User(username='adam', email='barnesheaton@gmail.com')
        u.launch_task('example', 42)

    return render_template('tasks.html', title='Tasks', form=form)
