import requests
import numpy as np

class DataParser:
    def __init__(self, ticker):
        self.ticker = ticker
        self.apiKey = self.getApiKey()

    def getApiKey(self):
        with open("api_key.txt") as file:
            return file.read()

    def getOHLC(self):
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + self.ticker + '&outputsize=full&apikey=' + self.apiKey
        r = requests.get(url)
        return r.json()

    def getSentiment(self):
        url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=' + self.ticker + '&apikey=' + self.apiKey
        r = requests.get(url)
        return r.json()

    def parseOHLC(self):    
        data = self.getOHLC()
        formattedData = []
        for day in data["Time Series (Daily)"]:
            newRow = []
            for category in data["Time Series (Daily)"][day]:
                newRow.append(data["Time Series (Daily)"][day][category])
            formattedData.append(newRow)
        formattedData = np.array(formattedData)
