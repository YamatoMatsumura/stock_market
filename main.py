from data import StockDataContainer
from neural_network import NeuralNetwork
from result_saver import saveResults

import numpy as np
import tensorflow as tf
import keras_tuner as kt


x = ['BNED', 'GOOG', 'NVDA', 'COIN']
STOCK_NAMES = ['SPOT']
EPOCHS = 150
TESTING_SIZE = 1
EARLY_STOP_PATIENCE = 20
BATCH_SIZE = 1
UPDATE_DATA = False
CREATE_NEW_MODEL = False

def main():
    for stock in STOCK_NAMES:
        # Initialize stock data container
        stockData = StockDataContainer(stock)
        if UPDATE_DATA:
            stockData.updateAllData()

        # Create neural network from data
        stockPredictor = NeuralNetwork(stockData)
        # Log trained data to avoid retraining from fresh every time
        stockPredictor.logData()

        # Get dataset and labels to inverse transform
        dataset = stockPredictor.getDataset()

        # Split into training and testing data
        dataset = list(dataset)

        # Grab only last couple elements for testing
        testingDataSet = dataset[-1 * TESTING_SIZE:]
        # Convert into lists since dataset is tuples of data, labels
        dataTestList = np.array([x[0].numpy() for x in testingDataSet])
        labelTestList = np.array([x[1].numpy() for x in testingDataSet])

        # Extract remaining data to use as training and split into data and labels
        trainingDataSet = dataset[:-1* TESTING_SIZE]
        dataTrainList = np.array([x[0].numpy() for x in trainingDataSet])
        labelTrainList = np.array([x[1].numpy() for x in trainingDataSet])

        # Convert to tf.data.Dataset to feed into neural network
        trainingDataSet = tf.data.Dataset.from_tensor_slices((dataTrainList, labelTrainList))
        testingDataSet = tf.data.Dataset.from_tensor_slices((dataTestList, labelTestList))

        # Initialize early stopping
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
        if CREATE_NEW_MODEL:

            # Initialize tuner
            tuner = kt.Hyperband(
                stockPredictor.getModel,
                objective='val_loss',
                max_epochs=50,
                factor=3,
                directory=f"data/{stockPredictor.ticker}",
                project_name=f"Hyperparam Tuning",

                #********************************
                # Temporary while adjusting params in tuner. Remove at very end 
                overwrite=True
                #********************************
            )

            # Preform hyperparam tuning
            tuner.search(trainingDataSet, epochs=EPOCHS, validation_data=(testingDataSet), callbacks=[earlyStopping])

            # Create model based on best hps
            bestHps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(bestHps)
            history = model.fit(trainingDataSet, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(testingDataSet), callbacks=[earlyStopping])

            # Keep track of model for result saving
            stockPredictor.model = model

        else:
            # Load model
            version = input("Select version to load (0 for old): ")
            if version == 0:
                model = tf.keras.models.load_model(f'data/{stockPredictor.ticker}/model.keras')
            model = tf.keras.models.load_model(f'data/{stockPredictor.ticker}/{str(version)}/model.keras')

            # Train model on new data
            # ****************************************************************
            # Make sure dataTrain and labelTrain only contain new data and not all the data so it doesn't model.fit on all the data again
            #*****************************************************************
            history = model.fit(trainingDataSet, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(testingDataSet), callbacks=[earlyStopping])

            # Keep track of model for result saving
            stockPredictor.model = model
        
        # Reduce data dimensions from 4D to 3D since indexing it made it 4D
        dataTestList = dataTestList.reshape(-1, dataTestList.shape[2], dataTestList.shape[3])
        # Reduce label dimensions from 3D to 2D since indexing it made it 3D
        labelTestList = labelTestList.reshape(-1, labelTestList.shape[2])

        # Make predictions
        predictions = model.predict(dataTestList)

        # Get best val loss from training history
        valLoss = history.history['val_loss']
        bestValLoss = min(valLoss)

        # Prepare metadata
        metadata = [TESTING_SIZE, bestValLoss]

        # Save results from training session
        saveResults(metadata, stockPredictor, predictions, labelTestList)


if __name__ == '__main__':
    main()