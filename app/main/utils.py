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

def p2f(x):
   return float(x.strip('%'))/100

def xor(lst1, lst2):
    lst3 = [value.lower() for value in lowerList(lst1) if value not in lowerList(lst2)]
    return lst3

def printLine(title=''):
    print(f"\n--------------------- {title} -----------------------\n")

def printData(title='', data=''):
    print(f"{title.replace('_', '').strip()} == [{data}]")

def getTickerList(file="tickers.csv", start=0, end=-1):
    df = pd.read_csv(file, names=["Symbol", "Description"])
    ticker_list = df["Symbol"].to_numpy()
    return ticker_list[start:end]

def getTickerString(file="tickers.csv", start=0, end=-1):
    df = pd.read_csv(file, names=["Symbol", "Description"])
    tickerList = df["Symbol"].to_numpy()
    tickerString = reduce(lambda a, b: a + " " + b, tickerList[start:end])
    return tickerString 