import numpy as np


def convert_dataset(dataset: list[str]) -> list[tuple[list[float], list[float]]]:
    converted_dataset = []

    for record in dataset:
        all_values = record.split(',')

        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(10) + 0.01
        targets[int(all_values[0])] = 0.99

        converted_dataset.append((inputs, targets))

    return converted_dataset
