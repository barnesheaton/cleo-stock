from app import app
from flask import render_template, current_app
from app.forms import QueueForm
from app.session import Session


@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    session = Session()
    form = QueueForm()

    if form.is_submitted():
        session.launch_task('example', 42)

    return render_template('index.html', title='Home', form=form, session=session)

