from app import app, db
from flask import render_template, current_app
from app.forms import QueueForm, UpdateStockDataForm
from app.session import Session


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    session = Session()
    updateForm = UpdateStockDataForm()

    if updateForm.is_submitted():
        session.launch_task('updateTickerTablesTask', period=updateForm.period.data, start=int(updateForm.start.data), end=int(updateForm.end.data))

    return render_template('index.html', title='Home', updateForm=updateForm, session=session)

