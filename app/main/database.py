import pandas as pd
import yfinance as yf
import numpy as np
from app import db
import random
import sys
import app.main.utils as utils
from app.models import StockModel
from sqlalchemy import Table, MetaData


class Database():
    connection = db.engine

    def __init__(self):
        self.connection = db.engine

    def getPomegranteTrainingData(self, tickerList, observation_period=50):
        sequences = []
        for ticker in tickerList:
            databaseHasTable = self.connection.dialect.has_table(self.connection, ticker.lower())
            if databaseHasTable:
                # build sequences
                df = self.getTickerData(ticker.lower()).dropna()
                observations_length = df.shape[0] - observation_period
                if observations_length >= 2:
                    observations = df.iloc[0:observation_period]['close'].to_numpy()
                    for index in range(1, observations_length):
                        observations = np.row_stack((observations, df.iloc[index : index + observation_period]['close'].to_numpy()))
                        # observations.append(df.iloc[index : index + observation_period]['close'].to_numpy())

                    utils.printLine(ticker)
                    print('# of observations :: ', observations.shape[0])
                    print('observation_period :: ', observations.shape[1])
                    # print(observations)
                    sequences.append(observations)

        print('sequences', sequences)
        return sequences
        
    def getTrainingData(self, tickerList):
        dataframe = pd.DataFrame()
        for ticker in tickerList:
            databaseHasTable = self.connection.dialect.has_table(self.connection, ticker.lower())
            if databaseHasTable:
                df = self.getTickerData(ticker.lower()).dropna()
                # TODO this check should eventually be based on the minimum rows needed to build a feature, w/ a fallback
                if df.shape[0] >= 21:
                    dataframe = pd.concat([dataframe, self.getTickerData(ticker.lower())])

        return dataframe

    def updateTickerTables(self, period, start=0, end=100):
        ticker_list = utils.getTickerList(start=start, end=end)
        ticker_string = utils.getTickerString(start=start, end=end)
        yf_df = yf.download(tickers=ticker_string, period=period, group_by="ticker")

        for ticker in ticker_list:
            df = yf_df.dropna() if (len(ticker_list) == 1) else yf_df[ticker].dropna()
            if df.shape[0] <= 10:
                continue

            if not self.connection.dialect.has_table(self.connection, ticker.lower()):
                self.createTickerTable(ticker.lower())

            self.updateTickerTable(ticker.lower(), df)

    def updateTickerTable(self, table, dataframe):
        dataframe = dataframe.rename(columns={"Adj Close": "adj_close"})
        dataframe.reset_index(level=0, inplace=True)
        dataframe.columns = dataframe.columns.str.lower()
        
        # Merge new YF data with exiwsting DB data, and remove duplicates
        old_df = pd.read_sql_table(table, con=self.connection)
        dataframe = pd.merge(old_df['date'], dataframe, how="right", indicator=True, on=["date"]).loc[lambda x:x['_merge'] == 'right_only']
        dataframe.drop(columns=['_merge'], inplace=True)
        dataframe.to_sql(table, con=self.connection, if_exists='append', index=False)

    def savePredictions(self, dataframe):
        dataframe.to_sql('predictions', con=self.connection, if_exists='append', index=False)

    def createTickerTable(self, table):
        if not self.connection.dialect.has_table(self.connection, table):
            metadata = MetaData(self.connection)
            Table(table, metadata,
                db.Column('date', db.Date(), primary_key=True, index=True, unique=True),
                db.Column('open', db.Float()),
                db.Column('high', db.Float()),
                db.Column('low', db.Float()),
                db.Column('close', db.Float()),
                db.Column('adj_close', db.Float()),
                db.Column('volume', db.Float()),
            )
            metadata.create_all()

    def getTickerTablesList(self, sample_percent=None, tickerString=None):
        tickers = utils.getTickerList()
        tables = self.connection.table_names()
        tickerTables = utils.intersection(tickers, tables)

        if sample_percent:
            k = len(tickerTables) * float(sample_percent) // 100
            indicies = random.sample(range(len(tickerTables)), int(k))
            sampledTickerTables = [tickerTables[i] for i in indicies]
            return sampledTickerTables

        elif tickerString:
            return utils.intersection(tickers, tickerString.replace(" ", "").strip(",").split(","))

        return tickerTables

    def getTickerData(self, table, limit=None):
        if limit:
            sql = f"""SELECT
                        *
                    FROM (
                        SELECT
                            *
                        FROM
                            {table}
                        ORDER BY
                            {table}.date DESC) AS {table}_temp
                    LIMIT {limit}
            """
            return pd.read_sql(sql=sql, con=self.connection).iloc[::-1]
        else:
            return pd.read_sql_table(table, self.connection)

    

    def getTickerDataToDate(self, table, date, days):
        if (days == 0 or days == 'max'):
            sql = f"""SELECT
                        *
                    FROM
                        {table}
                    WHERE
                        {table}.date <= '{date.strftime('%Y-%m-%d')}'
                    ORDER BY
                        {table}.date DESC
            """
        else:
            sql = f"""SELECT
                        *
                    FROM (
                        SELECT
                            *
                        FROM
                            {table}
                        WHERE
                            {table}.date <= '{date.strftime('%Y-%m-%d')}'
                        ORDER BY
                            {table}.date DESC
                        ) AS {table}_temp
                    LIMIT {days}
            """
        dataframe = pd.read_sql(sql=sql, con=self.connection)
        return dataframe.iloc[::-1]

    def getTickerDataAfterDate(self, table, date, days):
        if (days == 0 or days == 'max'):
            sql = f"""SELECT
                        *
                    FROM
                        {table}
                    WHERE
                        {table}.date >= '{date.strftime('%Y-%m-%d')}'
                    ORDER BY
                        {table}.date ASC
            """
        else:
            sql = f"""SELECT
                        *
                    FROM
                        {table}
                    WHERE
                        {table}.date >= '{date.strftime('%Y-%m-%d')}'
                    ORDER BY
                        {table}.date ASC
                    LIMIT {days}
            """
        dataframe = pd.read_sql(sql=sql, con=self.connection)
        return dataframe

    def getPlotData(self, plot_task_id, ticker):
        sql = f"""SELECT
                ticker, date, open, high, low, close, adj_close, volume
            FROM
                predictions
            WHERE
                predictions.task_id = {plot_task_id}
                AND predictions.ticker = '{ticker}'
        """
        dataframe = pd.read_sql(sql=sql, con=self.connection)
        return dataframe

    def getTickersInPlotTask(self, plot_task_id):
        sql = f"""SELECT
                    ticker
                FROM
                    predictions
                WHERE
                    predictions.task_id = {plot_task_id}
                    GROUP BY ticker
        """
        dataframe = pd.read_sql(sql=sql, con=self.connection)
        tickers = dataframe['ticker'].to_list()
        return tickers
