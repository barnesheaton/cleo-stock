from flask import render_template
from app.forms import UpdateStockDataForm
from app.session import Session


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    session = Session()
    updateForm = UpdateStockDataForm()

    if updateForm.is_submitted():
        session.launch_task('updateTickerTablesTask', period=updateForm.period.data, start=updateForm.start.data, end=updateForm.end.data)

    return render_template('index.html', title='Home', updateForm=updateForm, session=session)

