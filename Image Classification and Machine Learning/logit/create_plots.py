# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np

# plot the step function
plt.style.use("ggplot")
plt.figure()
plt.title("Step Function")
plt.plot([0.0, 2.3, 2.3, 5.0],[0.0, 0.0 ,1.0 ,1.0])
plt.ylim(-0.1,1.1)
plt.show()

# plot the first sigmoid function
x = np.arange(-6, 6, 0.1)
y = 1 / (1 + (np.e ** (-x)))
plt.style.use("ggplot")
plt.figure()
plt.plot(x, y)
plt.title("Sigmoid (Zoomed In)")
plt.xlim([-6, 6])
plt.show()

# plot the second sigmoid functionkkqq
x = np.arange(-50, 50, 0.1)
y = 1 / (1 + (np.e ** (-x)))
plt.style.use("ggplot")
plt.figure()
plt.plot(x, y)
plt.title("Sigmoid (Zoomed Out)")
plt.xlim([-50, 50])
plt.show()