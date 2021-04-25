import numpy as np

a = np.array([[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]])


b = a.T
b[0][0] = 0

print('a', a)
print('b', b)

