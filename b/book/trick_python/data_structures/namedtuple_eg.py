from collections import namedtuple
from sys import getsizeof
p1 = namedtuple('Point','x y z')(1,2,3)
p2 = (1,2,3)
print(p1)
print(p2)
print('sys.getsizeof p1 ', getsizeof(p1))
print('sys.getsizeof p2 ', getsizeof(p2))

