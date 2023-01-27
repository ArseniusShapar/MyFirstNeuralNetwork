def get_training_data(path='datasets/mnist_train.csv') -> list[str]:
    with open(path) as file:
        return [line.strip() for line in file.readlines()]


def get_testing_data(path='datasets/mnist_test.csv.csv') -> list[str]:
    with open(path) as file:
        return [line.strip() for line in file.readlines()]
