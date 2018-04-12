# Given an autocorrelation matrix, crops a circular region and then calculates
# the spatial periodicity by rotating the autocorrelation matrix in steps of 
# 6 degrees and computing the correlation

import math
import numpy as np
import scipy as sp
import scipy.misc
import skimage as sk
import skimage.transform 
import matplotlib.pyplot as plt

def calculate_spatial_periodicity(autocorrelationMatrix):
    maxGridScore = -math.inf
    maxThreshold = 0
    maxShiftedCircularMatrix = autocorrelationMatrix
    
    # Enhances matrices for smoother circle
    expand = 3
    autocorrelationMatrix = np.kron(autocorrelationMatrix, np.ones((expand, expand)))
    
    # Threshold can be any value between 0 and 1, but normally 0.1 is OK
    thresholdArray = np.arange(0, 0.4, 0.1)
    for threshold in thresholdArray:
        modifiedMatrix = np.copy(autocorrelationMatrix)
        modifiedMatrix[modifiedMatrix <= threshold] = 0
        modifiedMatrix[modifiedMatrix > threshold] = 1
        modifiedMatrix, numFeatures = sp.ndimage.measurements.label(modifiedMatrix)

        dim = autocorrelationMatrix.shape
        yDim = dim[0]
        xDim = dim[1]

        # Determines and identifies the center of the autocorrelation matrix
        yMatrixCenter = math.ceil(yDim / 2)
        xMatrixCenter = math.ceil(xDim / 2)
        if (modifiedMatrix[yMatrixCenter, xMatrixCenter] != 0):
            centerId = modifiedMatrix[yMatrixCenter, xMatrixCenter]
        else:
            # DOUBLE CHECK THAT THIS WORKS 
            rowAllPeaks, colAllPeaks = np.where(modifiedMatrix != 0)
            distanceFromCenter = np.sqrt(np.square(yMatrixCenter - rowAllPeaks) + np.square(xMatrixCenter - colAllPeaks))
            idx = np.where(distanceFromCenter == np.min(distanceFromCenter))
            centerId = modifiedMatrix[rowAllPeaks[idx]][colAllPeaks[idx]]

        # Gets center of central peak
        rowCenterPeak, colCenterPeak = np.where(modifiedMatrix == centerId)
        xCenterPeak = math.ceil((np.max(colCenterPeak) + np.min(colCenterPeak)) / 2)
        yCenterPeak = math.ceil((np.max(rowCenterPeak) + np.min(rowCenterPeak)) / 2)
        center = np.array([yCenterPeak, xCenterPeak])

        # Creates a matrix of all coordinates in peaks besides the central peak
        # First col is the row of the coordinate, second col is the col, third
        # col is the ID value of the coordinate in the modifiedMatrix (IDs which
        # peak the coordinate belongs to), and fourth col is the distance of the
        # coordinate to the center of the central peak
        rowPeaks, colPeaks = np.where((modifiedMatrix != 0) & (modifiedMatrix != centerId))
        coordinates = np.concatenate([np.reshape(rowPeaks, (len(rowPeaks), 1)), 
                                       np.reshape(colPeaks, (len(rowPeaks), 1)),
                                       np.reshape(modifiedMatrix[(rowPeaks, colPeaks)], (len(rowPeaks), 1)), 
                                       np.reshape(np.sqrt(np.square(yCenterPeak - rowPeaks) + np.square(xCenterPeak - colPeaks)), (len(rowPeaks), 1))], axis=1)
        sortedCoordinates = coordinates[np.argsort(coordinates[:, 3])]
        uniqueIdsCoords = np.sort(np.unique(sortedCoordinates[:, 2], return_index=True)[1])
        sortedUniqueIds = sortedCoordinates[:, 2][uniqueIdsCoords]
        if (sortedUniqueIds.size >= 6):
            peakIds = sortedUniqueIds[:6]
        else:
            peakIds = sortedUniqueIds

        # Gets coordinates of 6 closest fields and calculates the radius of the
        # circular area as the farthest distance from the center to any 
        # coordinate in the 6 fields
        circularPeaks = np.in1d(modifiedMatrix, peakIds).reshape(modifiedMatrix.shape)
        rowCircularPeaks, colCircularPeaks = np.where(circularPeaks)
        distances = np.sqrt(np.square(yCenterPeak - rowCircularPeaks) + np.square(xCenterPeak - colCircularPeaks))
        radius1 = np.max(distances)

        # Extracts circular area from autocorrelation matrix
        yMask1, xMask1 = np.ogrid[-yCenterPeak:yDim-yCenterPeak, -xCenterPeak:xDim-xCenterPeak]
        mask1 = xMask1**2 + yMask1**2 <= radius1**2
        
        # Calculates radius of circle to surround central peak
        centerDistances = np.sqrt(np.square(yCenterPeak - rowCenterPeak) + np.square(xCenterPeak - colCenterPeak))
        radius2 = np.max(centerDistances)
        yMask2, xMask2 = np.ogrid[-yCenterPeak:yDim-yCenterPeak, -xCenterPeak:xDim-xCenterPeak]
        mask2 = xMask2**2 + yMask2**2 > radius2**2
        
        # Gets mask with 2 circles to extrat information from rate map
        mask3 = mask1 & mask2
        mask3 = mask3.astype(float)
        mask3[np.where(mask3 == 0)] = np.nan
        circularMatrix = autocorrelationMatrix * mask3
        
        # Concatanates nan vectors horizontally and vertically to center the 
        # circular area for later crosscorrelation calculations
        horizontalShift = abs(xMatrixCenter - xCenterPeak)
        verticalShift = abs(yMatrixCenter - yCenterPeak)

        if (horizontalShift != 0):
            horizontalAdd = np.empty([yDim, horizontalShift])
            horizontalAdd[:] = np.nan
            if (xCenterPeak > xMatrixCenter):
                shiftedCircularMatrix = np.concatenate([circularMatrix, horizontalAdd], axis=1)
            else:
                shiftedCircularMatrix = np.concatenate([horizontalAdd, circularMatrix], axis=1)
        
        if (verticalShift != 0):
            verticalAdd = np.empty([verticalShift, xDim + horizontalShift])
            verticalAdd[:] = np.nan
            if (yCenterPeak > yMatrixCenter):
                shiftedCircularMatrix = np.concatenate([shiftedCircularMatrix, verticalAdd])
            else:
                shiftedCircularMatrix = np.concatenate([verticalAdd, shiftedCircularMatrix])
        
        gridScore = calculate_grid_score(shiftedCircularMatrix)
        if (gridScore > maxGridScore):
            maxGridScore = gridScore
            maxThreshold = threshold
            maxShiftedCircularMatrix = shiftedCircularMatrix
            
    rotations = np.reshape(np.arange(0, 366, 6), (61, 1))
    correlations = np.empty([61, 1])
    for i in range(61):
        correlations[i] = calculate_correlation(maxShiftedCircularMatrix, 6*i)
        
    return rotations, correlations, maxGridScore, maxShiftedCircularMatrix, maxThreshold
        
    
# Calculates the grid score of a cell by rotating the rate map of the cell and 
# computing the correlation between the rotated map and the original
def calculate_grid_score(rateMap):
    # Expected peak correlations of sinusoidal modulation
    correlation60 = calculate_correlation(rateMap, 60)
    correlation120 = calculate_correlation(rateMap, 120)
    
    # Expected trough correlations of sinusoidal modulation
    correlation30 = calculate_correlation(rateMap, 30)
    correlation90 = calculate_correlation(rateMap, 90)
    correlation150 = calculate_correlation(rateMap, 150)
    
    gridScore = min(correlation60, correlation120) - max(correlation30, correlation90, correlation150)
    
    return gridScore
    
    
# Calculates the correlation between a matrix and the matrix rotated by a 
# specified angle 
def calculate_correlation(matrix, angle):
    # How much the matrix was expanded by (look at line 24)
    expand = 3
    
    # Rotates the matrix. Also resolves the issue of an index having a value of 
    # 0.
    minValue = np.nanmin(matrix)
    tempMatrix = matrix - minValue + 1
    rotatedMatrix = skimage.transform.rotate(tempMatrix, angle)
    rotatedMatrix[rotatedMatrix == 0] = np.nan
    rotatedMatrix = rotatedMatrix + minValue - 1
    
    # Calculates the correlation of the original and rotated rate map
    row, col = matrix.shape
    matrixMask = np.copy(matrix)
    matrixMask[np.where(~np.isnan(matrixMask))] = 1
    rotatedMatrixMask = np.copy(rotatedMatrix)
    rotatedMatrixMask[np.where(~np.isnan(rotatedMatrixMask))] = 1
    newMatrix = matrix * rotatedMatrixMask
    newRotatedMatrix = rotatedMatrix * matrixMask
    
    sum1 = np.nansum(newMatrix * newRotatedMatrix)
    sum2 = np.nansum(newMatrix)
    sum3 = np.nansum(newRotatedMatrix)
    sum4 = np.nansum(newMatrix * newMatrix)
    sum5 = np.nansum(newRotatedMatrix * newRotatedMatrix)
    
    # Number of pixels in the rate map for which rate was estimated for both 
    # the original and rotated rate map
    n = np.where(~np.isnan(newMatrix))[0].shape[0]

    # Only estimate autocorrelation if n > 20 * expand pixels 
    if (n < 20 * expand):
        correlation = np.nan
    else:
        numerator = (n * sum1) - (sum2 * sum3)
        denominator = math.sqrt((n * sum4) - sum2**2) * math.sqrt((n * sum5) - sum3**2)
        correlation = numerator / denominator
    
    return correlation
    