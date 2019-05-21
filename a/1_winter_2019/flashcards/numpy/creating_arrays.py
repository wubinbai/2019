import numpy as np
a = np.array([1,2,3])
b = np.array([(1.5,2,3),(4,5,6)], dtype = float)
c = np.array([[(1.5,2,3),(4,5,6)],[(3,2,1),(4,5,6)]], dtype = float)

zeros = np.zeros((3,4))
ones = np.ones((2,3,4), dtype = np.int16)
arange = np.arange(10,25,5)
linspace = np.linspace(0,2,9)
full = np.full((2,2),7)
eye = np.eye(2)
random = np.random.random((2,2))
empty = np.empty((3,2))


