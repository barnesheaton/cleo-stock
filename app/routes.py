from app import app
from flask import render_template, current_app
from app.forms import QueueForm
from app.session import Session


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    session = Session()
    form = QueueForm()
    form2 = QueueForm()

    if form.is_submitted():
        session.launch_task('example', 42)

    if form2.is_submitted():
        session.populateBarsTable(period="1Y", tickers=False)

    return render_template('index.html', title='Home', form=form, form2=form2, session=session)

