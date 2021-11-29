import os
from app import app, db
from flask import render_template, current_app, url_for
from app.forms import QueueForm, UpdateStockDataForm, TrainModelForm, SimulateForm
from app.session import Session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, DecimalField

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

@app.route('/simulations', methods=['GET', 'POST'])
def simulations():
    session = Session()

    simulateForm = SimulateForm()

    if simulateForm.is_submitted():
        print("is_submitted")
        session.launch_task(
            'simulateTask',
            lookback_period=simulateForm.lookback,
            prediction_period=simulateForm.lookahead,
            start_date=simulateForm.start_date,
            end_date=simulateForm.end_date,
            principal=simulateForm.principal,
            diversification=simulateForm.diversification
        )

    return render_template('simulations.html', title='Simulate', simulateForm=simulateForm, session=session)
