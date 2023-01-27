import numpy as np
from scipy.special import expit


class NeuralNetwork:
    inodes: int
    hnodes: int
    onodes: int
    lr: float
    wih: np.ndarray
    who: np.ndarray
    new_weights: bool

    def __init__(self,
                 input_nodes: int,
                 hidden_nodes: int,
                 output_nodes: int,
                 learning_rate: float,
                 new_weights=False) -> None:

        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        self.new_weights = new_weights
        self.wih, self.who = self._getWeights()
        self.activation_function = expit

    def train(self, inputs_list: list[float], targets_list: list[float]) -> None:
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = self.wih @ inputs
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = self.who @ hidden_outputs
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = self.who.T @ output_errors

        self.who += self.lr * (output_errors * final_outputs * (1.0 - final_outputs)) @ np.transpose(hidden_outputs)

        self.wih += self.lr * np.dot((hidden_errors *
                                      hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list: list[float]) -> np.ndarray:
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = self.wih @ inputs
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = self.who @ hidden_outputs
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def loadWeights(self) -> tuple[np.ndarray, np.ndarray]:
        wih = np.load('wih.npy')
        who = np.load('who.npy')
        return wih, who

    def saveWeights(self) -> None:
        np.save('wih.npy', self.wih)
        np.save('who.npy', self.who)

    def generateWeights(self) -> tuple[np.ndarray, np.ndarray]:
        wih = np.random.normal(0.0, self.hnodes ** -0.5, (self.hnodes, self.inodes))
        who = np.random.normal(0.0, self.onodes ** -0.5, (self.onodes, self.hnodes))
        return wih, who

    def _getWeights(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            wih, who = self.loadWeights()
            if wih.shape == (self.hnodes, self.inodes) and who.shape == (self.onodes, self.hnodes) and self.new_weights:
                return wih, who
            else:
                return self.generateWeights()
        except FileNotFoundError:
            return self.generateWeights()
