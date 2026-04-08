import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.scatter([1,2,3,4], [2,3,4,5])
plt.savefig("test.png")