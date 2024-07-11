from data import StockDataContainer
from neural_network import NeuralNetwork
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


EPOCHS = 150
TEST_SIZE= 0.05
BATCH_SIZE = 1
UPDATE_DATA = False
RETRAIN_MODEL = True

def main():
    amazonData = StockDataContainer('AMZN')
    if UPDATE_DATA:
        amazonData.updateAllData()
    
    amazonNN = NeuralNetwork(amazonData)
    amazonNN.logData()
    dataset, scalarLabels = amazonNN.getDataLabels()

    # model = amazonNN.getModel()
    tuner = kt.BayesianOptimization(
        amazonNN.getModel,
        objective='val_loss',
        max_trials=15,
        num_initial_points=3,
        directory=f"data/{amazonNN.ticker}",
        project_name=f"Hyperparam Tuning"
    )

    data = np.concatenate(list(dataset.map(lambda x, y: x)))
    label = np.concatenate(list(dataset.map(lambda x, y: y)))

    dataTrain, dataTest, labelTrain, labelTest = train_test_split(data, label, test_size=TEST_SIZE, shuffle=False)

    if RETRAIN_MODEL:
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        tuner.search(x=dataTrain, y=labelTrain, epochs=EPOCHS, validation_data=(dataTest, labelTest), callbacks=[earlyStopping])
        print("search initialized")
        input()
        # model.fit(dataTrain, labelTrain, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(dataTest, labelTest), callbacks=[earlyStopping])
        # model.save('data/' + amazonNN.ticker + '/model.keras')
    else:
        model = tf.keras.models.load_model('data/' + amazonNN.ticker + '/model.keras')
    
    bestHps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f'Best hyperparameters: {bestHps.values}')

    model = tuner.hypermodel.build(bestHps)
    model.fit(dataTrain, labelTrain, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(dataTest, labelTest), callbacks=[earlyStopping])

    predictions = model.predict(dataTest)
    predictions = scalarLabels.inverse_transform(predictions)
    labelTest = scalarLabels.inverse_transform(labelTest)

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
        plt.savefig('data/' + amazonNN.ticker + '/graph.png')

        # Show plot
        plt.show()

if __name__ == '__main__':
    main()