import pandas as pd
import yfinance as yf
from app import db
import app.main.utils as utils
from sqlalchemy import Table, MetaData


class Database():
    connection = db.engine

    def __init__(self):
        self.connection = db.engine

    def updateTickerTables(self, period, start=0, end=100):
        ticker_list, ticker_string = utils.getTickerList(start=start, end=end, tickers=False)
        yf_df = yf.download(tickers=ticker_string, period=period, group_by="ticker")

        for ticker in ticker_list:
            df = yf_df.dropna() if (len(ticker_list) == 1) else yf_df[ticker].dropna()
            if df.shape[0] <= 10:
                continue

            if not self.connection.dialect.has_table(self.connection, ticker.lower()):
                self.createTickerTable(ticker.lower())

            self.updateTickerTable(ticker, df)

    def updateTickerTable(self, table, dataframe):
        dataframe = dataframe.rename(columns={"Adj Close": "adj_close"})
        dataframe.columns = dataframe.columns.str.lower()
        dataframe.to_sql(table, con=self.connection, if_exists='append', index_label="date")

    def createTickerTable(self, table):
        if not self.connection.dialect.has_table(self.connection, table):
            metadata = MetaData(self.connection)
            Table(table, metadata,
            db.Column(db.String(128), primary_key=True),
            db.Column(db.Float()),
            db.Column(db.Float()),
            db.Column(db.Float()),
            db.Column(db.Float()),
            db.Column(db.Float()),
            db.Column(db.Float()),
            )
            metadata.create_all()

    def readFromTickerTable(self, table):
        return pd.read_sql_table(table, self.connection)
