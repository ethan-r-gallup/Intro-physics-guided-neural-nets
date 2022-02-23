import numpy as np
import matplotlib.pyplot as plt

b = np.array([])
batches = np.array([])

for skipsize in range(1, 100, 2):
    # hyper parameters
    input_size = 12  # 28x28
    hidden_size = 20  # how many neurons per hidden layer
    num_out = 4  # digits 0-9
    num_epochs = 500
    gamma = .999
    learning_rate = .1

    mask = np.arange(100, 3100, skipsize)
    x = len(mask)
    b = np.append(b, x)
    print(len(mask))
    batches = np.append(batches, int(8.87509976e-10*x**3 + 2.74530619e-06*x**2 + 1.39164881e-03*x + 9.58001062))
    print(len(b), len(batches))
plt.plot(b, batches, 'bo')
plt.show()
