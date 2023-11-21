import nashpy as nash 
import numpy as np

p1 = np.array([
    [3,1],[0,2]
])

p2 = np.array([
    [2,1],[0,3]
])

rps = nash.Game(p1, p2)

eqs = rps.support_enumeration()


res = list(eqs)
len(res)



import nashpy as nash 
import numpy as np
p1 = np.array([[0.7]])
p2 = 1-p1
rps = nash.Game(p1, p2)

eqs = rps.support_enumeration()


res = list(eqs)



import nashpy as nash
import numpy as np
A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
rps = nash.Game(A)
B = - A
rps = nash.Game(A, B)
eqs = rps.support_enumeration()
list(eqs)

