import numpy as np

from convert_data import convert_dataset
from get_data import get_training_data, get_testing_data
from neural_network import NeuralNetwork


def main():
    # Initializing neural network
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    new_weights = True
    neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, new_weights)

    # Getting data for training
    training_dataset = get_training_data('datasets/mnist_train.csv')
    converted_training_dataset = convert_dataset(training_dataset)

    # Training neural network
    for inputs, targets in converted_training_dataset:
        neural_network.train(inputs, targets)
        neural_network.saveWeights()

    # Getting data for testing
    testing_dataset = get_testing_data('datasets/mnist_test.csv')
    converted_testing_dataset = convert_dataset(testing_dataset)

    # Testing neural network
    N = len(converted_testing_dataset)
    score = 0
    for i, record in enumerate(converted_testing_dataset):
        correct = int(testing_dataset[i][0])
        result = np.argmax(neural_network.query(record[0]))
        score += 1 if result == correct else 0

    print(f'\nEfficiency: {round(score / N * 100, 2)}%')


if __name__ == '__main__':
    main()
