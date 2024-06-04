import requests
import numpy as np
import pandas as pd
import json
import datetime as dt

class StockDataContainer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.apiKey = self._getApiKey()
        self.data = None

    def _getApiKey(self):
        with open('api_key.txt') as file:
            return file.read()
    
    def updateAllData(self):
        self.updateOHLCData()
        self.updateSentimentData()

    def updateOHLCData(self):
        # Request for all OHLC data
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + self.ticker + '&outputsize=full&apikey=' + self.apiKey
        r = requests.get(url)
        rawData = r.json()

        # Check and fix max api call errors
        if self._checkMaxAPICall(rawData):
            rawData = self._fixMaxAPICallFail(rawData, url)

        # #-------------------------------------------------
        # # Debug line, delete later
        # with open ('OHLC-AMZN.json', 'w') as file:
        #     json.dump(rawData, file)
        # #---------------------------------------------------
        # Format data into pandas df
        formattedData = self._parseOHLC(rawData)

        if self.data is None:
            self.data = formattedData
        else:
            self.data = pd.merge(self.data, formattedData, on='Date')

            
    def updateSentimentData(self):
        today = dt.datetime.now()
        if 'Sentiment' in self.data.columns.tolist():
            # Find most recent Sentiment and fill in up from there
            pass
        else:
            # Keep getting sentiment data until none found

            #Initialize empty dataframe
            columns = ['Date', 'Sentiment (Avg)', 'Number of Articles']
            df = pd.DataFrame(columns=columns)
            dayDelta = 365*2 + 80

            while True:
                # Initialize time frame to 24 hours
                day = dt.datetime.now() - dt.timedelta(days=dayDelta)
                timeTo = day.strftime('%Y%m%d') + 'T2359'
                timeFrom = day.strftime('%Y%m%d') + 'T0000'
                # Make API call
                url = ('https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
                       '&tickers=' + self.ticker + 
                       '&time_from=' + timeFrom + 
                       '&time_to=' + timeTo + 
                       '&apikey=' + self.apiKey)
                r = requests.get(url)
                rawData = r.json()

                # Check and fix max api call errors
                if self._checkMaxAPICall(rawData):
                    rawData = self._fixMaxAPICallFail(rawData, url)
                
                # Check for no more articles
                if self._checkSentimentError(rawData):
                    print("No more sentiment data")
                    print("Ended on: " + timeFrom + '-' + timeTo)
                    break

                # #--------------------------------------------------------
                # # Debug line, delete later
                # with open ('Sentiment-AMZN.json', 'w') as file:
                #     json.dump(rawData, file)
                # #---------------------------------------------------------

                # Format data into pandas df
                newRow = self._parseSentiment(rawData)
                df.loc[len(df)] = newRow

                # Adjust time frame back one day
                dayDelta += 1
            
            # add new sentiment data into self.data
            self.data = pd.merge(self.data, df, on='Date')

    def _parseOHLC(self, rawData):
        parsedData = [] # Holds date, OHLC, and volume data in a 2d list
        for day in rawData["Time Series (Daily)"]:
            newRow = []

            # Add formatted date data
            day = pd.to_datetime(day)
            day = day.strftime('%Y-%m-%d')
            newRow.append(day)

            # Add each OHLC and volume data
            for category in rawData["Time Series (Daily)"][day]:
                newRow.append(rawData["Time Series (Daily)"][day][category])
            parsedData.append(newRow)

        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Return pandas data frame
        return pd.DataFrame(parsedData, columns=columns)
    
    def _parseSentiment(self, data):
        # Grab time, average sentiment, and total article data
        row = []
        totalArticles = 0
        totalScore = 0
    
        # Check for no articles found
        if data['items'] == "0":
            return None

        # Format date data
        day = pd.to_datetime(data['feed'][0]['time_published'])
        day = day.strftime('%Y-%m-%d')
        row.append(day)

        # Grab total score and article count
        for article in data['feed']:
            for sentiment in article['ticker_sentiment']:
                if sentiment['ticker'] == self.ticker:
                    totalScore += float(sentiment['ticker_sentiment_score'])
                    totalArticles += 1
        row.append(totalScore / totalArticles)
        row.append(int(totalArticles))
        return row

    def _checkMaxAPICall(self, data):
        maxAPICallError = "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."
        if 'Information' in data.keys() or 'Note' in data.keys():
            for key in data:
                if maxAPICallError in data[key]:
                    print("Max API calls reached")
                    input("")
                    return True
        return False
    
    def _fixMaxAPICallFail(self, data, url):
        while self._checkMaxAPICall(data):
            r = requests.get(url)
            data = r.json()
        return data

    def _checkSentimentError(self, data):
        noArticlesError = "No articles found. Please adjust the time range or refer to the API documentation"
        if 'Information' in data.keys() or 'Note' in data.keys():
            for key in data:
                if noArticlesError in data[key]:
                    return True
        else:
            return False