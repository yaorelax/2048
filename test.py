import numpy as np
import matplotlib.pyplot as plt

configs = [[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1],
           [1, 1, 0, 0],
           [1, 1, 1, 0],
           [1, 1, 1, 1]]
name_list = [''.join(str(x) for x in config) for config in configs]
min_list = []
max_list = []
mean_list = []
var_list = []
x = list(range(len(configs)))
for config in configs:
    scores = list(range(1, 100))
    min_list.append(np.min(scores))
    max_list.append(np.max(scores))
    mean_list.append(np.mean(scores))
    var_list.append(np.var(scores))

plt.subplot(221)
plt.title('min')
plt.bar(x, min_list, tick_label=name_list)

plt.subplot(222)
plt.title('max')
plt.bar(x, max_list, tick_label=name_list)

plt.subplot(223)
plt.title('mean')
plt.bar(x, mean_list, tick_label=name_list)

plt.subplot(224)
plt.title('var')
plt.bar(x, var_list, tick_label=name_list)

plt.show()