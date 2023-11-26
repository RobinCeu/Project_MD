import numpy as np
import scipy.sparse as sparse

# Test DDXCentral
# x = np.array([-1.0,1.0,0.0])/2.0
# DDXCentral = sparse.diags(x, [-1, 1, 0], shape=(10, 10), dtype='float', format="csr")
# print(DDXCentral.toarray())
# y = DDXCentral.toarray()
# y[0][0] = 1.0
# y[0][1] = -1.0
# y[10-1][10-2] = -1.0
# y[10-1][10-1] = 1.0
# print(y)

# Test DDXBackward
# x = np.array([-1.0,1.0,0.0])
# DDXBackward = sparse.diags(x, [-1, 0, 1], shape=(10, 10), dtype='float', format="csr")
# print(DDXBackward.toarray())
# y = DDXBackward.toarray()
# y[0][0] = -1.0
# y[0][1] = 1.0
# print(y)

# Test DDXForward
# x = np.array([-1.0,1.0,0.0])
# DDXForward = sparse.diags(x, [0, 1, -1], shape=(10, 10), dtype='float', format="csr")
# print(DDXForward.toarray())
# y = DDXForward.toarray()
# y[10-1][10-2] = -1.0
# y[10-1][10-1] = 1.0
# print(y)

# Test D2DX2Central
# x = np.array([1.0,-2.0,1.0])
# D2DX2Central = sparse.diags(x, [-1,0,1], shape=(10, 10), dtype='float', format="csr")
# print(D2DX2Central.toarray())
# y = D2DX2Central.toarray()
# y[0][0] = 1.0
# y[0][1] = -2.0
# y[0][2] = 1.0
# y[10-1][10-3] = 1.0
# y[10-1][10-2] = -2.0
# y[10-1][10-1] = 1.0
# print(y)

# Test
""" changing name.toarray matrix, doenst change sparse.name matrix!!!!!, use sparse.name[0,0]"""
# x = np.array([1.0,-2.0,1.0])
# D2DX2Central = sparse.diags(x, [-1,0,1], shape=(10, 10), dtype='float', format="csr")
# print(D2DX2Central.toarray())
# D2DX2Central.toarray()[0][0] = 5
# #print(D2DX2Central)
# D2DX2Central[0,0] = 5.0
# print(D2DX2Central)

# Test D2DX2Backward
# x = np.array([1.0,-2.0,1.0])
# D2DX2Backward = sparse.diags(x, [0,-1,-2], shape=(10, 10), dtype='float', format="csr")
# print(D2DX2Backward.toarray())
# D2DX2Backward[0,0] = 1.0
# D2DX2Backward[0,1] = -2.0
# D2DX2Backward[0,2] = 1.0
# D2DX2Backward[1,0] = 0.0
# D2DX2Backward[1,1] = 1.0
# D2DX2Backward[1,2] = -2.0
# D2DX2Backward[1,3] = 1.0
# print(D2DX2Backward.toarray())

# Test D2DX2Forward
# x = np.array([1.0,-2.0,1.0])
# D2DX2Forward= sparse.diags(x, [0,1,2], shape=(10, 10), dtype='float', format="csr")
# print(D2DX2Forward.toarray())
# D2DX2Forward[10-2,10-4] = 1.0
# D2DX2Forward[10-2,10-3] = -2.0
# D2DX2Forward[10-2,10-2] = 1.0
# D2DX2Forward[10-2,10-1] = 0.0
# D2DX2Forward[10-1,10-3] = 1.0
# D2DX2Forward[10-1,10-2] = -2.0
# D2DX2Forward[10-1,10-1] = 1.0
# print(D2DX2Forward.toarray())

# # Dividing arrays
# a =np.array([1,2,3])
# b = np.array([2,2,2])
# print(a/b/12)

# Calling class
# # print(np.zeros(10)+2)
# class State:
#     def __init__(self,Grid):
               
#         #Scalar Fields
#         self.h=np.zeros(10) + Grid
# Vb = State(3)
# print(Vb.h)

# From array to diagonal matrix
# x = np.zeros(10) + 5
# xm = sparse.diags(x)
# print(xm.toarray())

# Multiplying Sparse matrices
# a = sparse.diags([1,2,3])
# b = sparse.diags([2,2,2])
# print(a*b)

#Multiplying sparse matrix with array
# a = sparse.diags([1,2,3])
# b = np.array([1,2,3])
# print(2*a*b)

# Multriplying arrays
# a = np.array([1,2,3])
# b = np.array([1,2,3])
# print(a*b)

# Call numpy array element
# a = np.array([1,2,3,4,5,6])
# print(a[1])

# max(array,0)
# a = np.array([-1,1,3,4,-5])
# b = np.maximum(a,0)
# print(b)


# ### Calling other classes ###
# class eerste_klasse:
#     def __init__(self, x):
#         self.x = x
#         self.random = 5


#     def vermenigvuldig(self,getal):
#         y = self.x*getal + self.random
#         print(y)

# test = eerste_klasse(2)
# x = test.vermenigvuldig(5)

# Array to sparse
# a = np.array([1,2,3,8,5,2])
# t = 8
# b = sparse.diags(a*t).toarray()
# print(b)

# Array mult with number
# a = np.array([1,2,3,5,6])
# print(a*3)

# Matrix data
# a = np.array([[1,2],[7,9]])
# print('hallo')
# b = a.data
# print(b)

# List of lists
# i = 5  # You can replace this with your desired number of lists
# my_list_of_lists = [[7] for _ in range(i)]
# print(my_list_of_lists)