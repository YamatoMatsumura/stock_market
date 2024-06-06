import pandas as pd
import numpy as np

class NeuralNetwork():
    def __init__(self, container):
        self.ticker = container.ticker
        self.data = container.data
    
    def getAllData(self):
        # Save trained data to trained_data.csv
        self.logData()

        # Delete Date Column since can't go into NN
        self.data = self.data.drop(columns=['Date'])

        # Convert labels and data to numpy array for NN
        labels = self.data.pop('Label').values
        data = self.data.values

        return data, labels

    def labelData(self):
        # See if current day stock price increased or decreased
        self.data['Difference'] = self.data['Close'] - self.data['Open']

        # Shift up by one to predict next day stock price increase or decrease
        self.data['Previous Day Difference'] = self.data['Difference'].shift(1)

        # if increased, label as 1, if decreased, label as 0
        self.data['Label'] = np.where(self.data['Previous Day Difference'] > 0, 1, 0)

        # Drop unneccesary columns
        self.data = self.data.drop(columns=['Difference' ,'Previous Day Difference'])

        # Remove today since won't have correct label since don't know if stock priced increased/decreased tomorrow
        self.data = self.data.drop(index=0)
        self.data = self.data.reset_index(drop=True)

    def convertData(self):
        # Convert the data to a numpy array to feed into NN
        pass
    def logData(self):
        # Check if existing data in trained_data
        try:
            # Load and add onto data
            df = pd.read_csv('data/' + self.ticker + '/trained_data.csv')
            self.data = pd.concat([df, self.data], ignore_index=True)       

            # Make sure dates are in right order
            self.data = self.data.sort_values(by='Date', ascending=False).reset_index(drop=True)

            # save to csv
            self.data.to_csv('data/' + self.ticker + '/trained_data.csv', index=False)   
        # If no pre-existing data  
        except (pd.errors.EmptyDataError, FileNotFoundError):
            #  Save everything to csv
            self.data.to_csv('data/' + self.ticker + '/trained_data.csv', index=False)

    def updateModel(self):
        # if model exists
            # load model
        # if no model already exists
            # Create new model
        
        # Feed all new_data into model
        # Maybe split into test and train? SO maybe make attributes of test and train data?

        # Save model at end
        pass
    def getModel(self):
        # Create the model
        pass