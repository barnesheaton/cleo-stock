from app import app
from flask import render_template, current_app
from app.forms import LoginForm, QueueForm
from app.models import User


@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    session = User()
    form = QueueForm()

    if form.is_submitted():
        session = User()
        session.launch_task('example', 42)

    return render_template('index.html', title='Home', form=form, session=session)

