import os
from cleoapp import app, db
from flask import render_template, current_app, url_for
from cleoapp.forms import QueueForm, UpdateStockDataForm, TrainModelForm
from cleoapp.session import Session

models_dir = os.path.join(app.instance_path, 'models')
os.makedirs(models_dir, exist_ok=True)
app.config['MODELS_DR'] = models_dir

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    session = Session()
    updateForm = UpdateStockDataForm()

    if updateForm.is_submitted():
        session.launch_task('updateTickerTablesTask', period=updateForm.period.data, start=int(updateForm.start.data), end=int(updateForm.end.data))

    return render_template('index.html', title='Home', updateForm=updateForm, session=session)


@app.route('/models', methods=['GET', 'POST'])
def models():
    session = Session()
    trainModelForm = TrainModelForm()

    if trainModelForm.is_submitted():
        session.launch_task('trainModelTask')

    return render_template('models.html', title='Models', trainModelForm=trainModelForm, session=session)
