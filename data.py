import requests
import pandas as pd
import datetime as dt
import os
import random

import vpn_script

RUN_SCRIPT = True
WINDOW_SIZE = 30
ANALYTICS_RANGE = '3year'
TECHNICAL_INDICATOR_WINDOW = 30
SENTIMENT_MISSING_PERCENT = 0.2

class StockDataContainer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.apiKey_AV = None
        self.apiKey_Polygon = None
        self.AVKeyCount = None
        self.data = None
        self.lastUpdated = self._getLastUpdated()
        self.scriptFirstTimeCalled = True

        # Create folder to store data
        self._createContainerFolder()

        # Grab all API Keys
        self._getApiKey()

        # Count number of AV keys to cycle through
        self._countAVKeys()
    
    def _createContainerFolder(self):
        if not os.path.exists('data/' + self.ticker):
            os.makedirs('data/' + self.ticker)

    def _getApiKey(self):
        with open('api_keys/Polygon.txt') as file:
            self.apiKey_Polygon = file.read()
        with open('api_Keys/AV/key0.txt') as file:
            self.apiKey_AV = file.read()
    
    def _countAVKeys(self):
        self.AVKeyCount = 0
        contents = os.listdir('api_keys/AV')
        for item in contents:
            itemPath = os.path.join('api_keys/AV', item)
            if os.path.isfile(itemPath):
                self.AVKeyCount += 1

    def _getLastUpdated(self):
        try:
            df = pd.read_csv('data/' + self.ticker + '/trained_data.csv')
            return df['Date'].iloc[0]
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return None
    
    def updateAllData(self):
        print("Updating all data...")

        # self.updateSentimentData()
        self.updateOHLCData()
        self.updateMeanData()
        self.updateReturnData()
        self.updateVarianceData()
        self.updateSTDDevData()
        self.updateMedianData()
        self.updateSMAData()
        self.updateEMAData()
        self.updateSTOCHData()
        self.updateRSIData()
        self.updateADXData()
        self.updateCCIData()
        self.updateAROONData()
        self.updateBBANDSData()
        self.updateADData()
        self.updateOBVData()
        # self.updateDateData()

        # Turn off vpn once done fetching data
        vpn_script.closeVpn(RUN_SCRIPT)


    def updateOHLCData(self):
        # Call Polygon API to get OHLC data
        endDate = dt.datetime.today().strftime('%Y-%m-%d')
        startDate = '2022-05-20' # Cutoff date is roughly 2 years before current date
        url = f'https://api.polygon.io/v2/aggs/ticker/{self.ticker}/range/1/day/{startDate}/{endDate}?apiKey={self.apiKey_Polygon}'
        response = requests.get(url)
        df = pd.DataFrame(response.json()['results'])

        # Format response
        df['Date'] = pd.to_datetime(df['t'] / 1000, unit='s')
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df.rename(columns= {
            'v': 'Volume',
            'vw': 'Volume Weighted Average',
            'o': 'Open',
            'c': 'Close',
            'h': 'High',
            'l': 'Low',
            'n': 'Transactions'
            })
        df.drop(columns=['t'], inplace=True)
        df = df.iloc[:, [7, 2, 4, 5, 3, 0, 1, 6]]

        # Reverse so newest date is at the top
        df = df.iloc[::-1]

        # Check if no data is currently stored
        if self.data is None or self.data.empty:
            self.data = df
        else:
            # Fill in OHLC data
            self.data = pd.concat([self.data, df], axis=0, join='outer')
     
    def updateSentimentData(self):
        # Initialize empty dataframe
        columns = ['Date', 'Sentiment (Avg)', 'Number of Articles']
        df = pd.DataFrame(columns=columns)

        # Initialize day delta to adjust time frame to search for
        dayDelta = 0

        while True:
            # Check halfway through if too many missing data points
            if dayDelta == 365:
                missingDataCount = (df['Number of Articles'] == 0).sum()
                if missingDataCount > len(df) * SENTIMENT_MISSING_PERCENT:
                    print(f"Missing more than {SENTIMENT_MISSING_PERCENT * 100}% of sentiment data, not including...")
                    return
            
            # Cutoff at 2 years (plus some more just to be safe) since OHLC date cuts off about 2 years before current date
            if dayDelta == 365*2 + 30:
                break

            # Initialize time frame to 24 hours
            day = dt.datetime.now() - dt.timedelta(days=dayDelta)
            timeTo = day.strftime('%Y%m%d') + 'T2359'
            timeFrom = day.strftime('%Y%m%d') + 'T0000'

            # Make API call
            url = ('https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
                    '&tickers=' + self.ticker + 
                    '&time_from=' + timeFrom + 
                    '&time_to=' + timeTo + 
                    '&apikey=' + self.apiKey_AV)
            r = requests.get(url)
            rawData = r.json()

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
        if self.data is None or self.data.empty:
            self.data = df
        else:
            # Fill in Sentiment data
            self.data = pd.concat([self.data, df], axis=0, join='outer')

    def updateDateData(self):
        # Split date into individual components to feed into network
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Day'] = self.data['Date'].dt.day
        self.data['Day of Week'] = self.data['Date'].dt.dayofweek

        # Make sure Date is formatted correctly
        self.data['Date'] = self.data['Date'].dt.strftime('%Y-%m-%d')
    
    def updateMeanData(self):
        self._updateAnalyticsData('MEAN', 'Mean')
    
    def updateMedianData(self):
        self._updateAnalyticsData('MEDIAN', 'Median')

    def updateReturnData(self):
        self._updateAnalyticsData('CUMULATIVE_RETURN', 'Cumulative Return')
    
    def updateVarianceData(self):
        self._updateAnalyticsData('VARIANCE', 'Variance')
    
    def updateSTDDevData(self):
        self._updateAnalyticsData('STDDEV', 'STDDEV')
    
    def updateSMAData(self):
        self._updateTechnicalIndicatorData('SMA')
    
    def updateEMAData(self):
        self._updateTechnicalIndicatorData('EMA')
    
    def updateSTOCHData(self):
        self._updateTechnicalIndicatorData('STOCH')
    
    def updateRSIData(self):
        self._updateTechnicalIndicatorData('RSI')
    
    def updateADXData(self):
        self._updateTechnicalIndicatorData('ADX')

    def updateCCIData(self):
        self._updateTechnicalIndicatorData('CCI')
    
    def updateAROONData(self):
        self._updateTechnicalIndicatorData('AROON')
    
    def updateBBANDSData(self):
        self._updateTechnicalIndicatorData('BBANDS')
    
    def updateADData(self):
        self._updateTechnicalIndicatorData('AD', 'Chaikin A/D')
    
    def updateOBVData(self):
        self._updateTechnicalIndicatorData('OBV')

    def _updateTechnicalIndicatorData(self, category, customCategory=False):
        # Make API Call
        url = ('https://www.alphavantage.co/query?' +
               'function=' + category +
               '&symbol=' + self.ticker +
               '&interval=daily' + 
               '&time_period=' + str(TECHNICAL_INDICATOR_WINDOW) + 
               '&series_type=close' + 
               '&apikey=' + self.apiKey_AV) 
        r = requests.get(url) 
        data = r.json() 

        # Check and fix max api call error
        if self._checkMaxAPICall(data):
            data = self._fixMaxAPICallFail(data, url)

        # Parse data
        if customCategory:
            data = data['Technical Analysis: ' + customCategory]
        else:
            data = data['Technical Analysis: ' + category]
        df = pd.DataFrame.from_dict(data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        # Check if no data is currently stored
        if self.data is None or self.data.empty:
            self.data = df
        else:
            # Fill in sentiment data
            self.data = pd.concat([self.data, df], axis=0, join='outer')

    
    def _updateAnalyticsData(self, category, categoryName, annualized=False):
        # Make API Call
        url = ('https://www.alphavantage.co/query?' + 
               'function=ANALYTICS_SLIDING_WINDOW' + 
               '&SYMBOLS=' + self.ticker + 
               '&RANGE=' + ANALYTICS_RANGE +
               '&INTERVAL=DAILY' + 
               '&OHLC=close'
               '&WINDOW_SIZE=' + str(WINDOW_SIZE) + 
               '&CALCULATIONS=' + category +
               '&apikey=' + self.apiKey_AV)
        r = requests.get(url)
        data = r.json()

        # Check and fix max api call error
        if self._checkMaxAPICall(data):
            data = self._fixMaxAPICallFail(data, url)

        # Adjusted name to take out annualized bit if needed
        if not annualized:
            adjustedName = category
        else:
            adjustedName = category[:-17]

        # Parse data
        data = data['payload']['RETURNS_CALCULATIONS'][category]['RUNNING_' + adjustedName][self.ticker]
        data = data.items()

        # Convert to pandas df
        df = pd.DataFrame(data, columns=['Date', categoryName])
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Check if no data is currently stored
        if self.data is None or self.data.empty:
            self.data = df
        else:
            # Fill in sentiment data
            self.data = pd.concat([self.data, df], axis=0, join='outer')


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
        if 'Information' in data.keys() or 'Note' in data.keys() or 'error' in data.keys():
            if 'error' in data.keys():
                return True
            for key in data:
                if maxAPICallError in data[key]:
                    return True
        return False
    
    def _fixMaxAPICallFail(self, data, url):
        attemps = 0 # Counter to retry fixing method if multiple failed attemps

        # If need new api key
        if 'error' in data.keys():
            self._getNewAVAPIKey()
            url = self._getNewAVURL(url)
        # If need to refresh VPN
        else:
            vpn_script.vpnRefreshScript(self.scriptFirstTimeCalled, RUN_SCRIPT)
            self.scriptFirstTimeCalled = False

        # Keep trying to make request until max api call doesn't happen
        while self._checkMaxAPICall(data):
            r = requests.get(url)
            data = r.json()
            attemps += 1

            # If just need a new api key
            if 'error' in data.keys():
                self._getNewAVAPIKey()
                url = self._getNewAVURL(url)
                continue

            # Refresh vpn again if tried to reconnect multiple times and not working
            if attemps == 3:
                vpn_script.vpnRefreshScript(self.scriptFirstTimeCalled, RUN_SCRIPT)
                attemps = 0
        return data

    def _checkSentimentError(self, data):
        noArticlesError = "No articles found. Please adjust the time range or refer to the API documentation"
        if 'Information' in data.keys() or 'Note' in data.keys():
            for key in data:
                if noArticlesError in data[key]:
                    return True
        else:
            return False
    
    def _getNewAVAPIKey(self):
        nextKeyPath = 'api_keys/AV/key' + str(random.randint(0, self.AVKeyCount - 1)) + '.txt'
        with open(nextKeyPath) as file:
            self.apiKey_AV = file.read()
    
    def _getNewAVURL(self, url):
        url = url[:-16]
        url += self.apiKey_AV
        return url
