import os
import json
from app import app
from flask import render_template
import math

from app.forms import DisplayPlotForm, PlotForm, UpdateStockDataForm, TrainModelForm, SimulateForm
from app.main.database import Database
from app.main.utils import printLine
from app.session import Session
from app.models import StockModel

from flask import request

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

models_dir = os.path.join(app.instance_path, 'models')
os.makedirs('uploads', exist_ok=True)
app.config['MODELS_DR'] = models_dir

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    session = Session()
    updateForm = UpdateStockDataForm()

    if updateForm.is_submitted():
        session.create_and_launch_task('updateTickerTablesTask', period=updateForm.period.data, start=int(updateForm.start.data), end=int(updateForm.end.data))

    return render_template('index.html', title='Home', updateForm=updateForm, session=session)


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
            model_type=trainModelForm.model_type.data,
            tickerString=trainModelForm.tickers.data
        )

    return render_template('models.html', title='Models', trainModelForm=trainModelForm, session=session)

@app.route('/simulations', methods=['GET', 'POST'])
def simulations():
    session = Session()
    simulateForm = SimulateForm()
    simulateForm.model.choices = modelList()

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
    (session, plotForm, displayPlotForm, _) = plotPageInit()

    if request.method == 'POST' and plotForm.validate():
        task = session.create_task('plotTask')
        session.launch_task(
            task=task,
            tickers=plotForm.tickers.data,
            task_id = task.id,
            model_id=int(plotForm.model.data),
            lookback_period=int(plotForm.lookback.data),
            prediction_period=int(plotForm.lookahead.data),
            limit=plotForm.limit.data
        )

    return render_template('plots.html', title='Plots', plotForm=plotForm, displayPlotForm=displayPlotForm, session=session)

@app.route('/plots/display', methods=['GET', 'POST'])
def displayPlots():
    (session, plotForm, displayPlotForm, graphJSON) = plotPageInit()
    tickerAccuracies = []

    if request.method == 'POST' and displayPlotForm.validate():
        database = Database()
        task_id = int(displayPlotForm.task.data)
        tickers = database.getTickersInPlotTask(task_id)
        prediction_period = database.getPlotTaskPredicitonPeriod(task_id)
        fig = make_subplots(rows=len(tickers), cols=1, row_titles=tickers)
        for ticker_index, ticker in enumerate(tickers):
            printLine(ticker)
            p_dataframe = database.getPlotData(task_id, ticker)
            start_date = p_dataframe.iloc[0]['date']
            verification_data = database.getTickerDataAfterDate(table=ticker, date=start_date, days=p_dataframe.shape[0])
            print(prediction_period, p_dataframe, verification_data)

            for si in range(0, prediction_period):
                index_predicted_closes = verification_data[( verification_data.index + si ) % prediction_period == 0]['close']
                index_verified_closes = p_dataframe[( verification_data.index + si ) % prediction_period == 0]['close']
                index_percentages = np.abs((index_predicted_closes - index_verified_closes) / index_verified_closes)
                index_accuracy = np.mean(np.abs(index_percentages))
                tickerAccuracies.append((ticker, si, index_accuracy))

            closePercentages = np.abs((p_dataframe['close'] - verification_data['close']) / verification_data['close'])
            accuracy = np.mean(np.abs(closePercentages))
            tickerAccuracies.append((ticker, 'total', accuracy))

            increments = math.floor(p_dataframe.shape[0] / prediction_period)   
            for inc in range(0, increments):
                start_index = inc * prediction_period
                end_index = (inc * prediction_period) + prediction_period

                fig.append_trace(go.Scatter(
                                x=p_dataframe[start_index:end_index]['date'],
                                y=p_dataframe[start_index:end_index]['close'],
                                mode='lines'),
                            index + 1, 1)


            fig.append_trace(go.Candlestick(x=verification_data['date'],
                                        open=verification_data['open'],
                                        high=verification_data['high'],
                                        low=verification_data['low'],
                                        close=verification_data['close']), ticker_index + 1, 1)

            # fig.append_trace(go.Line(x=p_dataframe['date'], y=p_dataframe['close']), index + 1, 1)
            # fig.append_trace(go.Line(x=verification_data['date'], y=verification_data['close']), index + 1, 1)
        
        fig.update_layout(width=1000, height=len(tickers) * 200)
        fig.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                dict(bounds=["sat", "mon"])  # hide weekends, eg. hide sat to before mon
            ]
        )
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('plots.html', title='Plots', plotForm=plotForm, displayPlotForm=displayPlotForm, session=session, graphJSON=graphJSON, tickerAccuracies=tickerAccuracies)

def plotPageInit():
    session = Session()
    graphJSON = None

    plotForm = PlotForm()
    plotForm.model.choices = modelList()

    displayPlotForm = DisplayPlotForm()
    displayPlotForm.task.choices = [(task.id, f"{task.name} - {task.id}") for task in session.get_task(name='plotTask')]

    return session, plotForm, displayPlotForm, graphJSON

def modelList():
    return [(model.id, f"{model.name} | {model.date.strftime('%m/%d/%Y, %H:%M:%S')} | obs. period: {model.observation_period} |") for model in StockModel.query.all()]
