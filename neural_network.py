import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
        self.scalarLabels = None
    
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

        # Reshape labels into 2D array since MinMaxSclar needs 2D array
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
        # Remove first sequenceLength labels since has to be skipped to create overlapping dataset
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

        # Return scalarLabels as well to use to inverse transform later
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

    def graphResults(self, predictions, labelTest):
        # Undo scaling on predictions and labels
        predictions = self.scalarLabels.inverse_transform(predictions)
        labelTest = self.scalarLabels.inverse_transform(labelTest)

        for i in range(len(labelTest)):
            rmse = np.sqrt(mean_squared_error(labelTest[i], predictions[i]))
            print("RMSE: ", rmse)
            # Plotting
            plt.figure(figsize=(10, 6))

            # Plot actual labels
            plt.plot(labelTest[i], linestyle='-', linewidth = 0.7, color='b', label='Actual', marker='o', markersize=1)

            # Plot predictions
            plt.plot(predictions[i], linestyle='--', linewidth = 0.7, color='r', label='Predicted', marker='o', markersize=1)

            # Customize plot
            plt.title('Actual vs. Predicted')
            plt.xlabel('Sample')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('data/' + self.ticker + '/graph.png')

            # Show plot
            plt.show()
        return
