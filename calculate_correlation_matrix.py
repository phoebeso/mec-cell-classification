import numpy as np
import scipy as sp

# dimensions of matrix 1 = dimensions of matrix 2 
def calculate_correlation_matrix(matrix1, matrix2=None):
    if (matrix2 == None):
        matrix2 = matrix1
    
    onesMatrix = np.ones(matrix1.shape)
    sumXY = sp.signal.correlate2d(matrix1, matrix2)
    sumX = sp.signal.correlate2d(matrix1, onesMatrix)
    sumY = sp.signal.correlate2d(onesMatrix, matrix2)
    xSq = sp.signal.correlate2d(np.square(matrix1), onesMatrix)
    ySq = sp.signal.correlate2d(onesMatrix, np.square(matrix2))
    n = calculate_n_matrix(matrix1)
    correlationMatrix = (n * sumXY - sumX * sumY) / ((np.sqrt(n * xSq - sumX**2) * np.sqrt((n+1) * ySq - sumY**2)) + 2**(-52))
    
    return correlationMatrix

# matrix 2 moves shifts across the surface of matrix 1
# if matrix 1 is MxN and matrix 2 is mxn, resulting matrix will be M+m-1xN+n-1
def calculate_n_matrix(matrix):
    row, col = matrix.shape
    baseCol = np.reshape(np.concatenate((np.arange(1, row+1), np.flip(np.arange(1, row), 0))), (2*row-1, 1))
    n = np.copy(baseCol)
    for i in range(2, 2*col):
        copy = np.copy(baseCol)
        if i <= col:
            newCol = i * copy
            n = np.concatenate((n, newCol), axis=1)
        else:
            newCol = (2*col-i) * copy
            n = np.concatenate((n, newCol), axis=1)
    
    return n
