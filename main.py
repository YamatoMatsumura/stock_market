from data import StockDataContainer
from neural_network import NeuralNetwork
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shap

EPOCHS = 100
TEST_SIZE= 0.1
BATCH_SIZE = 1
RETRAIN_MODEL = True
UPDATE_DATA = True

def main():
    amazonData = StockDataContainer('AMZN')
    if UPDATE_DATA:
        amazonData.updateAllData()

    amazonNN = NeuralNetwork(amazonData)
    dataset = amazonNN.getAllData()

    model = amazonNN.getModel()

    data = np.concatenate(list(dataset.map(lambda x, y: x)))
    label = np.concatenate(list(dataset.map(lambda x, y: y)))

    dataTrain, dataTest, labelTrain, labelTest = train_test_split(data, label, test_size=TEST_SIZE, shuffle=False)

    if RETRAIN_MODEL:
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model.fit(dataTrain, labelTrain, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(dataTest, labelTest), callbacks=[earlyStopping])
        model.save('data/' + amazonNN.ticker + '/model.keras')
    else:
        model = tf.keras.models.load_model('data/' + amazonNN.ticker + '/model.keras')

    predictions = model.predict(dataTest)

    total = 0
    for i in range(len(labelTest)):
        rmse = np.sqrt(mean_squared_error(labelTest[i], predictions[i]))
        total += rmse
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
    print(f'Average RMSE: {total / i}')

if __name__ == '__main__':
    main()