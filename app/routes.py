import os
import json
from app import app
from flask import render_template

from app.forms import DisplayPlotForm, PlotForm, UpdateStockDataForm, TrainModelForm, SimulateForm
from app.main.database import Database
from app.session import Session
from app.models import StockModel
from app.main.utils import printLine, printData, xor

from flask import request

import plotly
import plotly.express as px
import plotly.graph_objects as go

models_dir = os.path.join(app.instance_path, 'models')
os.makedirs('uploads', exist_ok=True)
app.config['MODELS_DR'] = models_dir

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    session = Session()
    updateForm = UpdateStockDataForm()

    graphJSON = None

    if updateForm.is_submitted():
        session.create_and_launch_task('updateTickerTablesTask', period=updateForm.period.data, start=int(updateForm.start.data), end=int(updateForm.end.data))

    df = Database().getTickerData(table='aapl')
    fig = px.line(df, x='date', y="close")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', title='Home', updateForm=updateForm, session=session, env=app.config['FLASK_ENV'], graphJSON=graphJSON)


@app.route('/models', methods=['GET', 'POST'])
def models():
    session = Session()
    trainModelForm = TrainModelForm()

    if request.method == 'POST' and trainModelForm.validate():
        session.create_and_launch_task(
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
        session.create_and_launch_task(
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
    (session, plotForm, displayPlotForm, graphJSON) = plotPageInit()

    if request.method == 'POST' and plotForm.validate():
        task = session.create_task('plotTask')
        session.launch_task(
            task=task,
            tickers=plotForm.tickers.data,
            task_id = task.id,
            model_id=int(plotForm.model.data),
            lookback_period=int(plotForm.lookback.data),
            prediction_period=int(plotForm.lookahead.data),
        )

    return render_template('plots.html', title='Plots', plotForm=plotForm, displayPlotForm=displayPlotForm, session=session)

@app.route('/plots/display', methods=['GET', 'POST'])
def displayPlots():
    (session, plotForm, displayPlotForm, graphJSON) = plotPageInit()

    if request.method == 'POST' and displayPlotForm.validate():

        database = Database()
        printLine('displayPlots')
        task_id = int(displayPlotForm.task.data)
        p_dataframe = database.getPlotData(task_id)
        tickers = database.getTickersInPlotTask(task_id)
        print(p_dataframe)
        print(tickers)
        tick = tickers[0]
        # df = database.getTickerData(table='aapl')
        verification_data = database.getTickerData(table=tick)
        print(verification_data)
        fig = px.line(verification_data, x='date', y="close")
        fig = fig.add_trace(go.Line(x = p_dataframe['date'], y = p_dataframe['close']))
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('plots.html', title='Plots', plotForm=plotForm, displayPlotForm=displayPlotForm, session=session, graphJSON=graphJSON)

def plotPageInit():
    session = Session()
    graphJSON = None

    plotForm = PlotForm()
    plotForm.model.choices = [(model.id, f"{model.name} - {model.date}") for model in StockModel.query.all()]

    displayPlotForm = DisplayPlotForm()
    displayPlotForm.task.choices = [(task.id, f"{task.name} - {task.id}") for task in session.get_all_tasks()]

    return session, plotForm, displayPlotForm, graphJSON