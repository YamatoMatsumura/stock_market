import data
from data import StockDataContainer
from neural_network import NeuralNetwork
import result_saver as saver
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import keras_tuner as kt


STOCK_NAMES = ['BNED', 'COIN', 'GOOG', 'META', 'MSFT', 'NVDA', 'SPOT']
EPOCHS = 70
EARLY_STOP_PATIENCE = 30
BATCH_SIZE = 3

UPDATE_DATA = False

CREATE_NEW_MODEL = False  # Does Hyperparm tuning & model.fit
TESTING = False  # Loads previous hyperparm tuning session
LOADING_MODEL = False # Loads previous model
TESTING_NEW_MODEL = True  # Creates model in nn.getManualModel(). No Hyperparm tuning


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


        # Get dataset
        dataset = stockPredictor.getDataset()

        # Convert to list to help split into training and testing
        dataset = list(dataset)

        # Grab last element for testing
        testingDataset = dataset[-1:]

        # Convert into np array since dataset is tuples of data, labels
        testingData = np.array([x[0].numpy() for x in testingDataset])
        testingLabels = np.array([x[1].numpy() for x in testingDataset])

        # Extract remaining data to use as training and split into data and labels
        trainingDataset = dataset[:-1 * data.WINDOW_SIZE]  # Space out Window size amount to make sure no part of testing dataset is in training
        trainingData = np.array([x[0].numpy() for x in trainingDataset])
        trainingLabels = np.array([x[1].numpy() for x in trainingDataset])

        # Convert to tf.data.Dataset to feed into neural network
        trainingDataset = tf.data.Dataset.from_tensor_slices((trainingData, trainingLabels))
        testingDataset = tf.data.Dataset.from_tensor_slices((testingData, testingLabels))

        # Create new directory to house training session data
        dirPath = saver.createNewDir(stockPredictor)

        # Initialize early stopping
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
        if CREATE_NEW_MODEL:
            
            # Initialize tuner
            tuner = kt.Hyperband(
                stockPredictor.getModel,
                objective='loss',
                max_epochs=50,
                factor=3,
                directory=dirPath,
                project_name=f"Hyperparam Tuning",

                # ********************************
                # Temporary while adjusting params in tuner. Remove at very end
                overwrite=True
                # ********************************
            )

            # Preform hyperparam tuning
            tuner.search(trainingDataset, epochs=EPOCHS, callbacks=[earlyStopping])

            # Create model based on best hps
            bestHps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(bestHps) 

            # Fit model on training data
            history = model.fit(trainingDataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[earlyStopping])

            # Keep track of model for result saving
            stockPredictor.model = model

        elif TESTING:
            
            # Choose version to load
            version = input("Select tuning session to load: ")

            # Initialize tuner from correct version
            tuner = kt.Hyperband(
                stockPredictor.getModel,
                objective='loss',
                max_epochs=50,
                factor=3,
                directory=f'data/{stockPredictor.ticker}/{str(version)}',
                project_name=f"Hyperparam Tuning",
            )

            # Reload previous tuner hyperparams
            tuner.reload()

            # Create model based on best hps
            bestHps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(bestHps)

            # Fit model on training data
            history = model.fit(trainingDataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[earlyStopping])

            # Keep track of model for result saving
            stockPredictor.model = model

        elif LOADING_MODEL:
            
            # Load model
            version = input("Select version to load (0 for old): ")

            if version == 0:
                model = tf.keras.models.load_model(f'data/{stockPredictor.ticker}/model.keras')
            model = tf.keras.models.load_model(f'data/{stockPredictor.ticker}/{str(version)}/model.keras')

            # Train model on new data
            # ****************************************************************
            # Make sure dataTrain and labelTrain only contain new data and not all the data so it doesn't model.fit on all the data again
            #*****************************************************************
            history = model.fit(trainingDataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[earlyStopping])

            # Keep track of model for result saving
            stockPredictor.model = model
        
        elif TESTING_NEW_MODEL:
            model = stockPredictor.getManualModel()

            history = model.fit(trainingDataset, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(testingDataset))

            # Extract the loss values
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']

            # Plot the learning curves
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Learning Curves')
            plt.legend()
            plt.savefig(f'{dirPath}/loss.png')
            # plt.show()

            stockPredictor.model = model
        

        # Reduce data dimensions from 4D to 3D since indexing dataset made list versions 4D
        testingData = testingData.reshape(-1, testingData.shape[2], testingData.shape[3])
        # Grab first element in case testing size > 1 since only graphing first set
        testingData = testingData[0]
        # Expand back to 3D since indexing first element made it 2D
        testingData = np.expand_dims(testingData, axis=0)

        # Reduce label dimensions from 3D to 2D since indexing it made it 3D
        testingLabels = testingLabels.reshape(-1, testingLabels.shape[2])
        # Repeat same first element + expanding back to 3D for labels
        testingLabels = testingLabels[0]
        testingLabels = np.expand_dims(testingLabels, axis=0)

        # Make predictions
        predictions = model.predict(testingData)

        # Get best val loss from training history
        valLoss = history.history['loss']
        bestValLoss = min(valLoss)

        # Prepare metadata
        metadata = [bestValLoss]

        # Save results from training session
        saver.saveResults(metadata, stockPredictor, predictions, testingLabels, dirPath)


if __name__ == '__main__':
    main()