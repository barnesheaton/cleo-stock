from flask import current_app
from app import db
import redis
import rq
from app.models import Task
import yfinance as yf
from app import utils

class Session():
    def launch_task(self, name, *args, **kwargs):
        rq_job = current_app.task_queue.enqueue('app.tasks.' + name, *args, **kwargs)
        task = Task(id=rq_job.get_id(), name=name)
        db.session.add(task)
        db.session.commit()
        return task

    def populateBarsTable(self, tickers, period, *args, start=0, end=100, **kwargs):
        ticker_list, ticker_string  = utils.getTickerList(start=start, end=end, tickers=tickers)
        yf_df = yf.download(tickers=ticker_string, period=period, group_by="ticker")

        for ticker in ticker_list:
            df = yf_df.dropna() if (len(ticker_list) == 1) else yf_df[ticker].dropna()
            if df.shape[0] <= 10:
                continue
            df.insert(0, "Ticker", ticker, True)
            df = df.rename(columns={"Adj Close": "adj_close"})
            df.columns = df.columns.str.lower()
            df.to_sql('bar', con=db.engine, if_exists='append', index_label="date")


    def get_all_tasks(self):
        return Task.query.all()

    def get_tasks_in_progress(self):
        return Task.query.filter_by(complete=False).all()

    def get_tasks_completed(self):
        return Task.query.filter_by(complete=True).all()

    def get_task(self, name):
        return Task.query.filter_by(name=name).first()
