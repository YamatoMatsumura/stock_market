import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import date
import os

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
        hpUnits1 = hp.Choice('units: 1', values=[4,8,16,32,64])
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
            hpUnits2 = hp.Choice('units: 2', values=[4,8,16,32])
            dropout2 = hp.Choice('Dropout: 2', values=[0.0, 0.2])
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                units=hpUnits2,
                return_sequences=True,
                recurrent_dropout=dropout2
            )))

        # Layer 3
        skipRegular = hp.Boolean('Skip optional regular')
        dropout3 = hp.Choice('Dropout: 3', values=[0.0, 0.2])
        if not skipRegular:
            hpUnits3 = hp.Choice('units: 3', values=[4,8,16,32])
            model.add(tf.keras.layers.LSTM(
                units=hpUnits3,
                return_sequences=True,
                recurrent_dropout=dropout3
            ))

        # Layer 4
        hpUnits4 = hp.Choice('units: 4', values=[4,8,16,32])
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

    def graphResults(self, predictions, labelTest, testSize, model):
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

            # Initialize metadata
            metadata = {
                'Date: ': str(date.today().strftime('%m/%d/%Y')),
                'Testing Size: ': str(testSize)
            }

            # Count number of graphs already existing to not overwrite previous ones
            fileCount = 1
            for root, dirs, files in os.walk(f'data/{self.ticker}'):
                for file in files:
                    if file.endswith('.png'):
                        fileCount += 1
            
            # Save plot
            plt.savefig(f'data/{self.ticker}/graph{fileCount}notes.png')

            # Write metadata to seperate file
            with open(f'data/{self.ticker}/graph{fileCount}.txt', 'w') as file:
                for key, value in metadata.items():
                    file.write(f'{key}: {value} \n')
                
                file.write('\n' + '-'*40)
                file.write('Model Summary')
                file.write('-'*40 + '\n')
                
                # Iterate through the layers and write to file the specific attributes
                for layer in model.layers:
                    # Check if the layer is Bidirectional
                    if isinstance(layer, tf.keras.layers.Bidirectional):
                        # Extract the wrapped LSTM layer
                        lstm_layer = layer._layers[0]
                        
                        if isinstance(lstm_layer, tf.keras.layers.LSTM):
                            file.write('Bidirectional LSTM Layer\n')
                            file.write(f'Units: {lstm_layer.units}\n')
                            file.write(f'Recurrent Dropout: {lstm_layer.recurrent_dropout}\n')
                            file.write('-' * 40 + '\n')
                    # Check if the layer is an LSTM layer
                    elif isinstance(layer, tf.keras.layers.LSTM):
                        file.write('LSTM Layer\n')
                        file.write(f'Units: {layer.units}\n')
                        file.write(f'Recurrent Dropout: {layer.recurrent_dropout}\n')
                        file.write('-' * 40 + '\n')
                    # Check if the layer is a Dense layer
                    elif isinstance(layer, tf.keras.layers.Dense):
                        file.write(f'Output Layer\n')
                        file.write(f'Units: {layer.units}\n')
                        file.write('-' * 40 + '\n')

            # Show plot
            plt.show()
        return
