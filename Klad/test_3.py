import numpy as np

# Creating a NumPy matrix
matrix = np.matrix([[1, 2], [3, 4]])

# Accessing the data attribute
matrix_data = matrix.data
print(matrix_data)

# Efficient Implementation for 1D csr type FD matrix
#   do not Change implementation below!
def SetDirichletLeft(M):
    M.data[[0,1,2]]=[1.0, 0.0, 0.0]

def SetDirichletRight(self,M):
    M.data[[-3,-2,-1]]=[0.0, 0.0, 1.0]

def SetNeumannLeft(self,M):
    M.data[[0,1,2]]=self.BackwardStencilD

def SetNeumannRight(self,M):
    M.data[[-3,-2,-1]]=self.ForwardStencilD
SetDirichletLeft(matrix)