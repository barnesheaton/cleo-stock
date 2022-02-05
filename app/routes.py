import os
from app import app
from flask import render_template
from app.forms import PlotForm, UpdateStockDataForm, TrainModelForm, SimulateForm
from app.session import Session
from app.models import StockModel
from flask import request

models_dir = os.path.join(app.instance_path, 'models')
os.makedirs('uploads', exist_ok=True)
app.config['MODELS_DR'] = models_dir

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    session = Session()
    updateForm = UpdateStockDataForm()

    if updateForm.is_submitted():
        session.launch_task('updateTickerTablesTask', period=updateForm.period.data, start=int(updateForm.start.data), end=int(updateForm.end.data))

    return render_template('index.html', title='Home', updateForm=updateForm, session=session, env=app.config['FLASK_ENV'])


@app.route('/models', methods=['GET', 'POST'])
def models():
    session = Session()
    trainModelForm = TrainModelForm()

    if request.method == 'POST' and trainModelForm.validate():
        session.launch_task(
            'trainModelTask',
            model_name=trainModelForm.name.data,
            model_description=trainModelForm.description.data,
            observation_period=int(trainModelForm.observation_period.data),
            sample_percent=trainModelForm.sample_percent.data,
            tickerString=trainModelForm.tickers.data
        )

    return render_template('models.html', title='Models', trainModelForm=trainModelForm, session=session)

@app.route('/simulations', methods=['GET', 'POST'])
def simulations():
    session = Session()
    simulateForm = SimulateForm()
    simulateForm.model.choices = [(model.id, f"{model.name} - {model.date}") for model in StockModel.query.all()]

    if request.method == 'POST' and simulateForm.validate():
        session.launch_task(
            'simulateTask',
            model_id=int(simulateForm.model.data),
            lookback_period=int(simulateForm.lookback.data),
            prediction_period=int(simulateForm.lookahead.data),
            start_date=simulateForm.start_date.data,
            end_date=simulateForm.end_date.data,
            principal=int(simulateForm.principal.data),
            diversification=int(simulateForm.diversification.data)
        )

    return render_template('simulations.html', title='Simulate', simulateForm=simulateForm, session=session)

@app.route('/plots', methods=['GET', 'POST'])
def plots():
    session = Session()
    plotForm = PlotForm()
    plotForm.model.choices = [(model.id, f"{model.name} - {model.date}") for model in StockModel.query.all()]

    if request.method == 'POST' and plotForm.validate():
        session.launch_task(
            'plotTask',
            ticker=plotForm.tickers.data,
            model_id=int(plotForm.model.data),
            lookback_period=int(plotForm.lookback.data),
            prediction_period=int(plotForm.lookahead.data),
        )

    return render_template('plots.html', title='Simulate', plotForm=plotForm, session=session)
