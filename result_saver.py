import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import date
import tensorflow as tf


def saveResults(metadata, neuralNetwork, predictions, labelTest):
    ticker = neuralNetwork.ticker

    # Count number of graphs already existing to not overwrite previous ones
    for root, dirs, files in os.walk(f'data/{ticker}'):
        dirCount = len(dirs)
        # Break after counting dirs in root directory
        break

    # Adjust if directory already exists
    if os.path.exists(f'data/{ticker}/{dirCount}'):
        dirCount += 1
    
    # Create directory to house this training session's data
    os.makedirs(f'data/{ticker}/{dirCount}')

    dirPath = f'data/{ticker}/{dirCount}'

    # Graph results
    graphResults(dirPath, neuralNetwork, predictions, labelTest)

    # Save metadata about session
    saveMetadata(metadata, dirPath, neuralNetwork)

    # Save keras model
    neuralNetwork.model.save(dirPath + '/model.keras')

def graphResults(dirPath, neuralNetwork, predictions, labelTest):

    # Undo scaling on predictions and labels
    predictions = neuralNetwork.scalarLabels.inverse_transform(predictions)
    labelTest = neuralNetwork.scalarLabels.inverse_transform(labelTest)

    for i in range(len(labelTest)):
        rmse = np.sqrt(mean_squared_error(labelTest[i], predictions[i]))
        print(rmse)

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

        # Save plot
        plt.savefig(f'{dirPath}/graph.png')


def saveMetadata(metadata, dirpath, neuralNetwork):
    model = neuralNetwork.model
    testSize = metadata[0]
    bestValLoss = metadata[1]

    # Initialize metadata
    metadata = {
        'Date: ': str(date.today().strftime('%m/%d/%Y')),
        'Testing Size: ': str(testSize),
        'Best Val Loss: ': str(bestValLoss),
        'Sequence Length: ': neuralNetwork.sequenceLength,
        'Predicting N Days: ': neuralNetwork.nDays
    }

    # Write metadata to seperate file
    with open(f'{dirpath}/notes.txt', 'w') as file:
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