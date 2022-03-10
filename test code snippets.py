import numpy as np
A = np.matrix([[1,0],[0,2]])
C = np.diag(A)

B = np.random.randn(2,1)
print(A*B)
print(np.random.normal(0,C,C.shape))

