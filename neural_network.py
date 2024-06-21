import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEQUENCE_LENGTH = 10
BATCH_SIZE = 3
N_DAYS = 30

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
        labels = self.data.pop('Close').values
        data = self.data.values

        # Scale data to help with fitting
        scalar = StandardScaler()
        data = scalar.fit_transform(data)

        self.numFeatures = data.shape[1]

        # Create overlapping dataset
        self.sequenceLength = SEQUENCE_LENGTH
        self.batchSize = BATCH_SIZE
        labels = labels[self.sequenceLength:] # Remove first sequenceLength labels since won't need
        newLabels = []
        for i in range(len(labels) - N_DAYS + 1):
            newLabels.append(labels[i : i + N_DAYS])
        
        newLabels = np.array(newLabels)

        labels = labels.reshape(-1, 1) # Reshape to 2D labels since data is 3D. Basically just a 2D array where each row just has one label

        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data,
            newLabels,
            self.sequenceLength,
            batch_size = self.batchSize
        )
        return dataset

    def logData(self):
        # Check if existing data in trained_data
        try:
            # Load and add onto data
            df = pd.read_csv('data/' + self.ticker + '/trained_data.csv')
            self.data = pd.concat([df, self.data], ignore_index=True)

            # Merge duplicate date days and fill in data by cross referencing rows
            self.data = self.data.groupby('Date').apply(lambda x: x.ffill().bfill().iloc[0]).reset_index(drop=True)

            # Get rid of all rows with missing data
            self.data = self.data.replace('', np.nan).dropna(axis=0, how='any').reset_index(drop=True)

            # Make sure no duplicates
            self.data.drop_duplicates(inplace=True)       

            # Make sure dates are in right order
            self.data = self.data.sort_values(by='Date', ascending=False).reset_index(drop=True)

            # save to csv
            self.data.to_csv('data/' + self.ticker + '/trained_data.csv', index=False)   
        # If no pre-existing data  
        except (pd.errors.EmptyDataError, FileNotFoundError, OSError):
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

        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(self.sequenceLength, self.numFeatures))))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True)))
        model.add(tf.keras.layers.LSTM(units=64))
        model.add(tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l1(l1=0.01), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))

        # Output layer
        model.add(tf.keras.layers.Dense(N_DAYS))

        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
            metrics=["mean_absolute_error"]
        )
        return model
