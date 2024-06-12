import pandas as pd
import numpy as np
import tensorflow as tf

class NeuralNetwork():
    def __init__(self, container):
        self.ticker = container.ticker
        self.data = container.data
        self.sequenceLength = None
        self.batchSize = None
        self.numFeatures = None
    
    def getAllData(self):
        # Save trained data to trained_data.csv
        self.logData()

        # Reverse data so trains from oldest to newest
        self.data = self.data.iloc[::-1]

        # Delete Date Column since can't go into NN
        self.data = self.data.drop(columns=['Date'])

        # Convert labels and data to numpy array for NN
        # labels = self.data.pop('Label').values
        labels = self.data.pop('Close').values
        self.data = self.data.drop(columns=['Label'])
        data = self.data.values

        self.numFeatures = data.shape[1]

        # Create overlapping dataset 
        self.sequenceLength = 5
        self.batchSize = 4
        labels = labels[self.sequenceLength:] # Remove first sequenceLength labels since won't need
        labels = labels.reshape(-1, 1) # Reshape to 2D labels since data is 3D. Basically just a 2D array where each row just has one label

        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data,
            labels,
            self.sequenceLength,
            batch_size = self.batchSize
        )

        return dataset
    

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
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(self.sequenceLength, self.numFeatures)))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model