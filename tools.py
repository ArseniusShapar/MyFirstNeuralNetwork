import matplotlib.pyplot as plt
import numpy as np


def show_image(record: str) -> None:
    all_values = record.split(',')
    image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()
