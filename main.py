from data import StockDataContainer
from neural_network import NeuralNetwork
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split



EPOCHS = 150
TEST_SIZE= 0.05
EARLY_STOP_PATIENCE = 20
BATCH_SIZE = 1
UPDATE_DATA = True
CREATE_NEW_MODEL = True

def main():

    # Initialize stock data container
    stockData = StockDataContainer('BNED')
    if UPDATE_DATA:
        stockData.updateAllData()

    # Create neural network from data
    stockPredictor = NeuralNetwork(stockData)
    # Log trained data to avoid retraining from fresh every time
    stockPredictor.logData()
    # Get dataset and labels to inverse transform
    dataset = stockPredictor.getDataset()

    # Format data and labels to train test split
    data = np.concatenate(list(dataset.map(lambda x, y: x)))
    label = np.concatenate(list(dataset.map(lambda x, y: y)))

    # Split data and labels into training and testing data
    dataTrain, dataTest, labelTrain, labelTest = train_test_split(data, label, test_size=TEST_SIZE, shuffle=False)

    if CREATE_NEW_MODEL:
        # Initialize early stopping
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)

        # Initialize tuner
        tuner = kt.BayesianOptimization(
            stockPredictor.getModel,
            objective='val_loss',
            max_trials=30,
            num_initial_points=5,
            directory=f"data/{stockPredictor.ticker}",
            project_name=f"Hyperparam Tuning"
        )

        # Preform hyperparam tuning
        tuner.search(x=dataTrain, y=labelTrain, epochs=EPOCHS, validation_data=(dataTest, labelTest), callbacks=[earlyStopping])

        # Create model based on best hps
        bestHps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(bestHps)
        model.fit(dataTrain, labelTrain, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(dataTest, labelTest), callbacks=[earlyStopping])

        # Save model
        model.save('model.h5')
    else:
        # Load model
        model = tf.keras.models.load_model('data/' + stockPredictor.ticker + '/model.keras')

        # Train model on new data
        # ****************************************************************
        # Make sure dataTrain and labelTrain only contain new data and not all the data so it doesn't model.fit on all the data again
        #*****************************************************************
        model.fit(dataTrain, labelTrain, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(dataTest, labelTest), callbacks=[earlyStopping])


    predictions = model.predict(dataTest)

    # Compare predictions vs actual values to determine accuracy
    stockPredictor.graphResults(predictions, labelTest)


if __name__ == '__main__':
    main()