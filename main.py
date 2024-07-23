from data import StockDataContainer
from neural_network import NeuralNetwork
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split


STOCK_NAME = 'BNED'
EPOCHS = 150
TESTING_SIZE = 2
EARLY_STOP_PATIENCE = 20
BATCH_SIZE = 1
UPDATE_DATA = True
CREATE_NEW_MODEL = True

def main():
    # Initialize stock data container
    stockData = StockDataContainer(STOCK_NAME)
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

    if CREATE_NEW_MODEL:
        # Initialize early stopping
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)

        # Initialize tuner
        tuner = kt.Hyperband(
            stockPredictor.getModel,
            objective='val_loss',
            max_epochs=60,
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
        model.fit(trainingDataSet, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(testingDataSet), callbacks=[earlyStopping])

        # Save model
        model.save(f'data/{stockPredictor.ticker}/model.keras')
    else:
        # Load model
        model = tf.keras.models.load_model('data/' + stockPredictor.ticker + '/model.keras')

        # Train model on new data
        # ****************************************************************
        # Make sure dataTrain and labelTrain only contain new data and not all the data so it doesn't model.fit on all the data again
        #*****************************************************************
        model.fit(trainingDataSet, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(testingDataSet), callbacks=[earlyStopping])


    predictions = model.predict(dataTestList)

    # Compare predictions vs actual values to determine accuracy
    stockPredictor.graphResults(predictions, labelTestList)


if __name__ == '__main__':
    main()