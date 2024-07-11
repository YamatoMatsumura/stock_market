import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEQUENCE_LENGTH = 30
BATCH_SIZE = 1
N_DAYS = 30

class NeuralNetwork():
    def __init__(self, container):
        self.ticker = container.ticker
        self.data = container.data
        self.sequenceLength = None
        self.batchSize = None
        self.numFeatures = None
    
    def getDataLabels(self):

        # Reverse data so trains from oldest to newest
        self.data = self.data.iloc[::-1]

        # Delete Date Column since can't go into NN
        self.data = self.data.drop(columns=['Date'])

        # Convert labels and data to numpy array for NN
        labels = self.data.pop('Close').values
        data = self.data.values

        # Scale data to help with fitting
        scalarData = MinMaxScaler()
        scalarLabels = MinMaxScaler()
        data = scalarData.fit_transform(data)

        # Reshape labels into 2D array since MinMaxSclar needs 2D array
        labels = labels.reshape(-1, 1)
        labels = scalarLabels.fit_transform(labels)
        # Reshape labels back into 1D array
        labels = labels.flatten()

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
            batch_size = self.batchSize,
        )
        return dataset, scalarLabels

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

    def updateModel(self):
        # if model exists
            # load model
        # if no model already exists
            # Create new model

        # Feed all new_data into model
        # Maybe split into test and train? SO maybe make attributes of test and train data?

        # Save model at end
        pass
    def getModel(self, hp):
        model = tf.keras.Sequential()

        # Layer 1
        hpUnits1 = hp.Choice('units: 1', values=[8,12,16,32,64])
        hpRecurrentDropouts1 = hp.Float('recurrent dropout: 1', min_value=0.0, max_value=0.5, step=0.1)
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=hpUnits1,
            return_sequences=True,
            input_shape=(self.sequenceLength, self.numFeatures),
            recurrent_dropout=hpRecurrentDropouts1
        )))

        # Layer 2
        skipBidirectional = hp.Boolean('Skip optional Bidirectional')
        if not skipBidirectional:
            hpUnits2 = hp.Choice('units: 2', values=[4,8,12,16,32])
            hpRecurrentDropouts2 = hp.Float('recurrent dropout: 2', min_value=0.0, max_value=0.5, step=0.1)
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                units=hpUnits2,
                return_sequences=True,
                recurrent_dropout=hpRecurrentDropouts2
            )))

        # Layer 3
        skipRegular = hp.Boolean('Skip optional regular')
        if not skipRegular:
            hpUnits4 = hp.Choice('units: 3', values=[2,4,6,8,12,16])
            hpRecurrentDropouts4 = hp.Float('recurrent dropout: 3', min_value=0.0, max_value=0.5, step=0.1)
            model.add(tf.keras.layers.LSTM(
                units=hpUnits4,
                return_sequences=True,
                recurrent_dropout=hpRecurrentDropouts4
            ))

        # Layer 4
        hpUnits3 = hp.Choice('units: 4', values=[2,4,6,8,12,16])
        hpRecurrentDropouts3 = hp.Float('recurrent dropout: 4', min_value=0.0, max_value=0.5, step=0.1)
        model.add(tf.keras.layers.LSTM(
            units=hpUnits3,
            recurrent_dropout=hpRecurrentDropouts3
        ))

        # Output Layer
        model.add(tf.keras.layers.Dense(N_DAYS))

        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
            metrics=["mean_absolute_error"]
        )

        return model


        # model = tf.keras.Sequential()

        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        #     units=32, 
        #     return_sequences=True, 
        #     input_shape=(self.sequenceLength, self.numFeatures),
        #     recurrent_dropout=0.2)))

        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=16, return_sequences=True, recurrent_dropout=0.2)))

        # # model.add(tf.keras.layers.LSTM(units=64, return_sequences=True, recurrent_dropout=0.2))

        # model.add(tf.keras.layers.LSTM(units=8, recurrent_dropout=0.2))

        # # model.add(tf.keras.layers.Dense(4, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.1), activation='relu'))

        # # Output layer
        # model.add(tf.keras.layers.Dense(N_DAYS))

        # model.compile(
        #     optimizer="adam",
        #     loss="mean_squared_error",
        #     metrics=["mean_absolute_error"]
        # )
        # return model
