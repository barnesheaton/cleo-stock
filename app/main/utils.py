import pandas as pd
from pathlib import Path
from functools import reduce
import numpy as np

def lowerList(list):
    result = list
    for i in range(len(list)):
        result[i] = result[i].lower()
    return result

def intersection(lst1, lst2):
    lst3 = [value.lower() for value in lowerList(lst1) if value in lowerList(lst2)]
    return lst3

def printLine(title=''):
    print(f"\n--------------------- {title} -----------------------\n")


def loadModelSettings(model_name):
    df = pd.read_csv(f"models/{model_name}.settings.csv")
    dictionary = df.to_dict(orient="index")
    return dictionary[0]


def saveModelSettings(settings, model_name):
    output_file = "settings.csv"
    output_dir = Path(f"models/{model_name}")
    df = pd.DataFrame(settings, index=[0])
    df.to_csv(output_dir / output_file)


def getTickerList(filename="tickers.csv", start=0, end=2700, tickers=False):
    df = pd.read_csv(filename, names=["Symbol", "Description"])
    ticker_list = df["Symbol"].to_numpy()
    ticker_list = ticker_list[start:end] if tickers == False else np.array(
        tickers)
    ticker_string = reduce(lambda a, b: a + " " + b, ticker_list)
    return ticker_list, ticker_string
