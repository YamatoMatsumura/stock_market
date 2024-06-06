import requests
import pandas as pd
import datetime as dt

import vpn_script

RUN_SCRIPT = True

class StockDataContainer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.apiKey = self._getApiKey()
        self.data = None
        self.lastUpdated = self._getLastUpdated()
        self.scriptFirstTimeCalled = True

    def _getApiKey(self):
        with open('api_key.txt') as file:
            return file.read()
    
    def _getLastUpdated(self):
        try:
            df = pd.read_csv('data/' + self.ticker + '/trained_data.csv')
            return df['Date'].iloc[0]
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return None
    
    def updateAllData(self):
        print("Updating all data...")

        self.updateOHLCData()
        self.updateSentimentData()
        self.updateDateData()

        # Turn off vpn once done fetching data
        vpn_script.closeVpn(RUN_SCRIPT)


    def updateOHLCData(self):
        # Request for all OHLC data
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + self.ticker + '&outputsize=full&apikey=' + self.apiKey
        r = requests.get(url)
        rawData = r.json()

        # Check and fix max api call errors
        if self._checkMaxAPICall(rawData):
            rawData = self._fixMaxAPICallFail(rawData, url)

        # Format data into pandas df
        df = self._parseOHLC(rawData)

        # Check if only need portion of OHLC data
        if self.lastUpdated is not None:
            df = df[df['Date'] > self.lastUpdated]
        
        # Make sure OHLC and volume data are all numeric
        df['Open'] = pd.to_numeric(df['Open'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['Close'] = pd.to_numeric(df['Close'])
        df['Volume'] = pd.to_numeric(df['Volume'])


        # Check if no data is currently stored
        if self.data is None:
            self.data = df
        else:
            # Fill in OHLC data
            self.data = pd.merge(self.data, df, on='Date')
     
    def updateSentimentData(self):
        # Initialize empty dataframe
        columns = ['Date', 'Sentiment (Avg)', 'Number of Articles']
        df = pd.DataFrame(columns=columns)

        # Initialize day delta to adjust time frame to search for
        dayDelta = 0

        while True:
            #---------------------------------------
            # Debug line, delete later
            if dayDelta == 20:
                break
            #--------------------------------------

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

            # with open('Sentiment-AMZN.json', 'w') as file:
            #     json.dump(rawData, file)

            # Check for max api call error
            if self._checkMaxAPICall(rawData):
                rawData = self._fixMaxAPICallFail(rawData, url)
            
            # Check for end of sentiment data
            if self._checkSentimentError(rawData):
                # Break out of loop since no more sentiment data available
                break

            # Format data into pandas df
            newRow = self._parseSentiment(rawData)
            # Handle case when no articles found for that day
            if newRow is None:
                newRow = [(dt.datetime.now() - dt.timedelta(days=dayDelta)).strftime('%Y-%m-%d'), 0, 0]
            df.loc[len(df)] = newRow

            # Adjust time frame back one day
            dayDelta += 1

            # Check if only need up to a certain day
            if self.lastUpdated == (dt.datetime.now() - dt.timedelta(days=dayDelta)).strftime('%Y-%m-%d'):
                break

        # Delete rows with missing data
        df = df[df['Number of Articles'] != 0]

        # Check if no data is currently stored
        if self.data is None:
            self.data = df
        else:
            # Fill in sentiment data
            self.data = pd.merge(self.data, df, on='Date')
       
    def updateDateData(self):
        # Split date into individual components to feed into network
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Day'] = self.data['Date'].dt.day
        self.data['Day of Week'] = self.data['Date'].dt.dayofweek

        # Make sure Date is formatted correctly
        self.data['Date'] = self.data['Date'].dt.strftime('%Y-%m-%d')

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
                    return True
        return False
    
    def _fixMaxAPICallFail(self, data, url):
        vpn_script.vpnRefreshScript(self.scriptFirstTimeCalled, RUN_SCRIPT)
        self.scriptFirstTimeCalled = False
        attemps = 0
        while self._checkMaxAPICall(data):
            r = requests.get(url)
            data = r.json()
            attemps += 1

            # Refresh vpn again if tried to reconnect multiple times and not working
            if attemps == 3:
                vpn_script.vpnRefreshScript(self.scriptFirstTimeCalled, RUN_SCRIPT)
        print("API call error resolved")
        return data

    def _checkSentimentError(self, data):
        noArticlesError = "No articles found. Please adjust the time range or refer to the API documentation"
        if 'Information' in data.keys() or 'Note' in data.keys():
            for key in data:
                if noArticlesError in data[key]:
                    return True
        else:
            return False
        
    
