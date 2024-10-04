import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


SEQUENCE_LENGTH = 30  # Sequence length for data set
BATCH_SIZE = 1  # Batch size for data set
N_DAYS = 30  # Controls how many days into the future to predict


class NeuralNetwork():
    def __init__(self, container):
        self.ticker = container.ticker
        self.data = container.data
        self.sequenceLength = None
        self.batchSize = None
        self.numFeatures = None
        self.scalarLabels = None
        self.model = None
        self.nDays = N_DAYS
    
    def getDataset(self):

        # Reverse data so trains from oldest to newest
        self.data = self.data.iloc[::-1]

        # Delete Date Column since can't go into NN
        self.data = self.data.drop(columns=['Date'])

        # Convert labels and data to numpy array for NN
        labels = self.data.pop('Close').values
        data = self.data.values

        # Scale data to help with fitting
        scalarData = MinMaxScaler()
        data = scalarData.fit_transform(data)

        # Reshape labels into 2D array since StandardScalar needs 2D array
        self.scalarLabels = MinMaxScaler()
        labels = labels.reshape(-1, 1)
        labels = self.scalarLabels.fit_transform(labels)
        # Reshape labels back into 1D array
        labels = labels.flatten()

        # Initialize/log attributes
        self.numFeatures = data.shape[1]
        self.sequenceLength = SEQUENCE_LENGTH
        self.batchSize = BATCH_SIZE

        # Create overlapping dataset
        # Remove first sequenceLength labels since predicting future closing price (i.e. day 1 of data has day N_DAYS+1 closing price)
        labels = labels[self.sequenceLength:] 
        windowedLabels = []
        for i in range(len(labels) - N_DAYS + 1):
            windowedLabels.append(labels[i : i + N_DAYS])
        
        windowedLabels = np.array(windowedLabels)

        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data,
            windowedLabels,
            self.sequenceLength,
            batch_size = self.batchSize,
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
        except:
            # Merge duplicate date days and fill in data by cross referencing rows
            self.data = self.data.groupby('Date').apply(lambda x: x.ffill().bfill().iloc[0]).reset_index(drop=True)

            # Get rid of all rows with missing data
            self.data = self.data.replace('', np.nan).dropna(axis=0, how='any').reset_index(drop=True)

            # Make sure dates are in right order
            self.data = self.data.sort_values(by='Date', ascending=False).reset_index(drop=True)

            #  Save everything to csv
            self.data.to_csv('data/' + self.ticker + '/trained_data.csv', index=False)


    def getModel(self, hp):
        model = tf.keras.Sequential()

        # Layer 1
        hpUnits1 = hp.Choice('units: 1', values=[4,8,16])
        dropout1 = hp.Choice('Dropout: 1', values=[0.0,0.2])
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=hpUnits1,
            return_sequences=True,
            input_shape=(self.sequenceLength, self.numFeatures),
            recurrent_dropout=dropout1
        )))

        # Layer 2
        skipBidirectional = hp.Boolean('Skip optional Bidirectional')
        if not skipBidirectional:
            hpUnits2 = hp.Choice('units: 2', values=[4,8,16])
            dropout2 = hp.Choice('Dropout: 2', values=[0.0, 0.2])
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                units=hpUnits2,
                return_sequences=True,
                recurrent_dropout=dropout2
            )))

        # Layer 3
        skipRegular = hp.Boolean('Skip optional regular')
        hpUnits3 = hp.Choice('units: 3', values=[4,8])
        dropout3 = hp.Choice('Dropout: 3', values=[0.0, 0.2])
        if not skipRegular:
            model.add(tf.keras.layers.LSTM(
                units=hpUnits3,
                return_sequences=True,
                recurrent_dropout=dropout3
            ))

        # Layer 4
        hpUnits4 = hp.Choice('units: 4', values=[4,8])
        dropout4 = hp.Choice('Dropout: 4', values=[0.0, 0.2])
        model.add(tf.keras.layers.LSTM(
            units=hpUnits4,
            recurrent_dropout=dropout4
        ))

        # Output Layer
        model.add(tf.keras.layers.Dense(N_DAYS))

        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
            metrics=["mean_absolute_error"]
        )

        return model

    def getManualModel(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=16, return_sequences=True, recurrent_dropout=0.2)))
        model.add(tf.keras.layers.LSTM(units=8, recurrent_dropout=0.2))
        model.add(tf.keras.layers.Dense(N_DAYS))

        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
            metrics=["mean_absolute_error"]
        )

        return model
