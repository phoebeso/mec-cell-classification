import numpy as np
import scipy as sp
import math
import warnings

def calculate_2d_tuning_curve(variable_x, variable_y, fr, numBin, minVal, maxVal):
    xAxis = np.linspace(minVal, maxVal, numBin+1)
    yAxis = np.linspace(minVal, maxVal, numBin+1)

    tuningCurve = np.zeros((numBin, numBin))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(numBin):
            start_x = xAxis[i]
            stop_x = xAxis[i+1]

            for j in range(numBin):
                start_y = yAxis[j]
                stop_y = yAxis[j+1]

                if (i == numBin-1):
                    if (j == numBin-1):
                        tuningCurve[numBin-1-j, i] = np.mean(fr[(variable_x >= start_x) & (variable_x <= stop_x) & (variable_y >= start_y) & (variable_y <= stop_y)])
                    else:
                        tuningCurve[numBin-1-j, i] = np.mean(fr[(variable_x >= start_x) & (variable_x <= stop_x) & (variable_y >= start_y) & (variable_y < stop_y)])
                else:
                    if (j == numBin-1):
                        tuningCurve[numBin-1-j, i] = np.mean(fr[(variable_x >= start_x) & (variable_x < stop_x) & (variable_y >= start_y) & (variable_y <= stop_y)])
                    else:
                        tuningCurve[numBin-1-j, i] = np.mean(fr[(variable_x >= start_x) & (variable_x < stop_x) & (variable_y >= start_y) & (variable_y < stop_y)])

    # Fill in the NaNs with neighboring values
    for i in range(numBin):
        for j in range(numBin):
            if (math.isnan(tuningCurve[j, i])):
                right = tuningCurve[j, min(i+1, numBin-1)]
                left = tuningCurve[j, max(i-1, 0)]
                down = tuningCurve[min(j+1, numBin-1), i]
                up = tuningCurve[max(j-1, 0), i]
                
                ru = tuningCurve[max(j-1, 0), min(i+1, numBin-1)]
                lu = tuningCurve[max(j-1, 0), max(i-1, 0)]
                rd = tuningCurve[min(j+1, numBin-1), min(i+1, numBin-1)]
                ld = tuningCurve[min(j+1, numBin-1), max(i-1, 0)]
                
                if (np.all(np.isnan([right, left, down, up, ru, lu, rd, ld]))):
                    tuningCurve[j, i] = 0
                else:
                    tuningCurve[j, i] = np.nanmean([right, left, down, up, ru, lu, rd, ld])

    # Smooth the tuning curve
    h = gaussian_2d()
    smoothTuningCurve = sp.ndimage.correlate(tuningCurve, h, mode='constant')

    return tuningCurve, smoothTuningCurve

def gaussian_2d(shape=(3, 3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
