import pandas as pd
import yfinance as yf
from app import db
import sys
import app.main.utils as utils
from sqlalchemy import Table, MetaData


class Database():
    connection = db.engine

    def __init__(self):
        self.connection = db.engine

    def getTrainingData(self):
        ticker_list, _ = utils.getTickerList()
        dataframe = pd.DataFrame()
        for index, ticker in enumerate(ticker_list):
            db_has_table = self.connection.dialect.has_table(self.connection, ticker.lower())
            # Take every other ticker table to seperate training data from test data
            if (index % 2 == 0) & db_has_table:
                dataframe = pd.concat([dataframe, self.getTickerData(ticker.lower())])

        return dataframe

    def updateTickerTables(self, period, start=0, end=100):
        ticker_list, ticker_string = utils.getTickerList(start=start, end=end, tickers=False)
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

    def createTickerTable(self, table):
        print("createTickerTable")
        if not self.connection.dialect.has_table(self.connection, table):
            metadata = MetaData(self.connection)
            Table(table, metadata,
                db.Column('date', db.Date(), primary_key=True, index=True),
                db.Column('open', db.Float()),
                db.Column('high', db.Float()),
                db.Column('low', db.Float()),
                db.Column('close', db.Float()),
                db.Column('adj_close', db.Float()),
                db.Column('volume', db.Float()),
            )
            metadata.create_all()

    def getTickerTablesList(self):
        tickers, _ = utils.getTickerList()
        tables = self.connection.table_names()
        return utils.intersection(tickers, tables)

    def getTickerData(self, table):
        return pd.read_sql_table(table, self.connection)

    def getTickerDataToDate(self, table, date, days):
        if (days == 0):
            sql = f"""SELECT
                        *
                    FROM
                        aame
                    WHERE
                        {table}.date <= '{date.strftime('%Y-%m-%d %H:%M:%S.%f')}'
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
                            {table}.date <= '{date.strftime('%Y-%m-%d %H:%M:%S.%f')}'
                        ORDER BY
                            {table}.date DESC
                        )
                    LIMIT {days}
            """
        return pd.read_sql(sql=sql, con=self.connection)