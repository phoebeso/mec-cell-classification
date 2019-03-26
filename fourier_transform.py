# Calculates the Fourier Spectrogram of a given rate map and determines the main 
# components and polar distribution

# Adapted from Neural Representations of Location Composed of Spatially Periodic 
# Bands, Krupic et al 2012

# Performs 2D FFT to calculate the fourier spectrogram and max power for the cell

import math
import numpy as np
import scipy as sp
import scipy.signal as signal
import random
from calculate_2d_tuning_curve import calculate_2d_tuning_curve
from shuffle_rate_map import shuffle_rate_map

def fourier_transform(rateMap, meanFr, spiketrain, dt, posx, posy):
    adjustedRateMap = rateMap - meanFr
    adjustedRateMap[np.where(np.isnan(adjustedRateMap))] = 0 
    
    fourierSpectrogram = np.fft.fft2(adjustedRateMap, s=[256, 256])
    fourierSpectrogram = fourierSpectrogram / (meanFr * math.sqrt(rateMap.shape[0] * rateMap.shape[1]))
    fourierSpectrogram = np.fft.fftshift(fourierSpectrogram)
    fourierSpectrogram = np.absolute(fourierSpectrogram)
    
    maxPower = np.max(fourierSpectrogram)
    
    shiftedPowers = np.empty([150, 65536])
    for i in range(150):
        minShift = math.ceil(20 / dt)
        maxShift = len(spiketrain) - (20 / dt)
        randShift = int(round(minShift + random.uniform(0, 1) * (maxShift - minShift)))
        
        shiftedSpiketrain = np.roll(spiketrain, randShift)
        shiftedFiringRate = np.reshape(shiftedSpiketrain / dt, (len(shiftedSpiketrain), 1))
        
        unsmoothShiftedRateMap, smoothShiftedRateMap = calculate_2d_tuning_curve(posx, posy, shiftedFiringRate, 20, 0, 100)
        unsmoothShiftedRateMap = unsmoothShiftedRateMap - meanFr
        unsmoothShiftedRateMap[np.where(np.isnan(unsmoothShiftedRateMap))] = 0
        
        shiftedFourier = np.fft.fft2(unsmoothShiftedRateMap, s=[256, 256])
        shiftedFourier = shiftedFourier / (meanFr * math.sqrt(unsmoothShiftedRateMap.shape[0] * unsmoothShiftedRateMap.shape[1]))
        shiftedFourier = np.absolute(np.fft.fftshift(shiftedFourier))
        shiftedFourier = shiftedFourier.reshape(-1)
        shiftedPowers[i] = shiftedFourier
    
    sigPercentile = np.percentile(shiftedPowers, 95)
    if (maxPower > sigPercentile):
        isPeriodic = True
    else:
        isPeriodic = False
        
    threshold1 = np.percentile(shiftedPowers, 50)
    polarSpectrogram = fourierSpectrogram - threshold1
    polarSpectrogram[np.where(polarSpectrogram < 0)] = 0
    
    # main components is used to generate the polar graph of the average power along each
    # radial angle (aka the flower looking polar graph)
    threshold2 = np.max(polarSpectrogram) * 0.1
    mainComponents = np.copy(polarSpectrogram)
    mainComponents[np.where(mainComponents < threshold2)] = 0
    
    return fourierSpectrogram, polarSpectrogram, maxPower, isPeriodic

# Attempts to identify polar components by extracting connected areas from the polar
# spectrogram
def analyze_fourier(fourierSpectrogram, polarSpectrogram):
    dim = polarSpectrogram.shape
    yDim = dim[0]
    xDim = dim[1]
    
    modifiedPolar, numFeatures = sp.ndimage.measurements.label(polarSpectrogram)
    
    yCenter = math.ceil(yDim / 2) - 1
    xCenter = math.ceil(xDim / 2) - 1
    
    radius1, radius2, start = 0, 0, 0
    
    polarComponents = np.zeros(((256,256,0)))
    
    while (radius1 < 256 and radius2 < 256):
        for radius1 in range(start, 256):
            ySmallMask, xSmallMask = np.ogrid[-yCenter:yDim-yCenter, -xCenter:xDim-xCenter]
            smallMask = (xSmallMask**2 + ySmallMask**2 <= radius1**2) & (xSmallMask**2 + ySmallMask**2 >= start**2)
            circularPolarSpectrogram = polarSpectrogram * smallMask
            
            if np.count_nonzero(circularPolarSpectrogram):
                break
        
        if (radius1 == 255):
            break
        
        start = radius1
            
        for radius2 in range(start, 256):
            yBigMask, xBigMask = np.ogrid[-yCenter:yDim-yCenter, -xCenter:xDim-xCenter]
            bigMask = (xBigMask**2 + yBigMask**2 >= radius2**2) & (xBigMask**2 + yBigMask**2 <= (radius2+1)**2)
            circularPolarSpectrogram = polarSpectrogram * bigMask
            
            if (np.count_nonzero(circularPolarSpectrogram) == 0):
                break
            
        yMask, xMask = np.ogrid[-yCenter:yDim-yCenter, -xCenter:xDim-xCenter]
        mask = (xMask**2 + yMask**2 >= radius1**2) & (xMask**2 + yMask**2 <= radius2**2)
        circularPolarSpectrogram = polarSpectrogram * mask
        
        polarComponents = np.dstack((polarComponents, circularPolarSpectrogram))
        
        start = radius2 + 1
    
    return polarComponents

# Calculates the average power along every radial line from the center of the 
# polar spectrogram and identifies the local maxima of the average power 
# distribution
def analyze_polar_spectrogram(polarSpectrogram):
    dim = polarSpectrogram.shape
    yDim = dim[0]
    xDim = dim[1]
    
    threshold = np.max(polarSpectrogram) * 0.1
    mainComponents = np.copy(polarSpectrogram)
    mainComponents[np.where(mainComponents < threshold)] = 0
        
    rhos = np.zeros(180)
    count = np.zeros(180)
    
    # Only calculates the average power of the upper half of the polar 
    # spectrogram due to the symmetry of 2D FFT
    for i in range(int(yDim / 2)):
        for j in range(int(xDim)):
            y = 128.5 - i
            x = j - 128.5
            if (x > 0):
                if (y > 0):
                    theta = math.floor(math.degrees(math.atan(y/x)))
                else:
                    theta = math.floor(math.degrees(math.atan(y/x) + 2*math.pi))
            else:
                theta = math.floor(math.degrees(math.atan(y/x) + math.pi));
                
            rho = mainComponents[i, j]
            rhos[theta] += rho
            count[theta] += 1
    
    rhoMeanPower = rhos / count
    rhoMeanPower = np.concatenate((rhoMeanPower, rhoMeanPower))
    rhoMeanPower = np.convolve(rhoMeanPower, signal.gaussian(17,13), mode='same')
    
    localMaxima = signal.argrelextrema(rhoMeanPower, np.greater)[0]
    
    # Delete local maxima within 10 degrees of a local maxima with a higher 
    # average power
    cont = True
    i = 0
    while cont:
        currTheta = localMaxima[i]
        currRho = rhoMeanPower[currTheta]

        nextTheta = localMaxima[i+1]
        nextRho = rhoMeanPower[nextTheta]
        if (nextTheta - currTheta < 10):
            if (currRho > nextRho):
                localMaxima = np.delete(localMaxima, [i+1])
            else:
                localMaxima = np.delete(localMaxima, [i])
        else:
            i += 1
        
        if i == len(localMaxima) - 1:
            cont = False

    localMaxima = localMaxima[:int(len(localMaxima)/2)]
    localMaxima = (localMaxima * math.pi) / 180
    
    return rhoMeanPower, localMaxima

# Calculates the average power of rings of various radial sizes from the fourier 
# spectrogram
def analyze_fourier_rings(spectrogram, area):
    dim = spectrogram.shape
    yDim = dim[0]
    xDim = dim[1]
    
    yCenter = math.ceil(yDim / 2) - 1
    xCenter = math.ceil(xDim / 2) - 1
    
    # maxRadius = math.floor(math.sqrt((xDim/2)**2 + (yDim/2)**2))
    
    maxRadius = math.ceil((area / (4 * math.pi)) - 1)
    
    averageRingPower = np.zeros((maxRadius, 1))
    radii = np.arange(0, maxRadius)
    
    for innerRadius in range(maxRadius):
        outerRadius = math.ceil(math.sqrt((area / math.pi) + innerRadius**2))
        
        yMask, xMask = np.ogrid[-yCenter:yDim-yCenter, -xCenter:xDim-xCenter]
        mask = (xMask**2 + yMask**2 >= innerRadius**2) & (xMask**2 + yMask**2 <= outerRadius**2)
        spectrogramRing = spectrogram[np.where(mask)]
        
        averageRingPower[innerRadius] = np.nanmean(spectrogramRing)
    
    return averageRingPower, radii

# Spiketrain is randomly suffled 150 times and the max power of the shifted fourier is calculated.
# Polar spectrograms are found by subtracting the 50th percentile of the power of the randomly
# generated fourier spectrograms. 
def calculate_polar_threshold(meanFr, spiketrain, dt, posx, posy):
    shiftedPowers = np.empty([150, 65536])
    for i in range(150):
        minShift = math.ceil(20 / dt)
        maxShift = len(spiketrain) - (20 / dt)
        randShift = int(round(minShift + random.uniform(0, 1) * (maxShift - minShift)))
        
        shiftedSpiketrain = np.roll(spiketrain, randShift)
        shiftedFiringRate = np.reshape(shiftedSpiketrain / dt, (len(shiftedSpiketrain), 1))
        
        unsmoothShiftedRateMap, _ = calculate_2d_tuning_curve(posx, posy, shiftedFiringRate, 20, 0, 100)
        unsmoothShiftedRateMap = unsmoothShiftedRateMap - meanFr
        unsmoothShiftedRateMap[np.where(np.isnan(unsmoothShiftedRateMap))] = 0
        
        shiftedFourier = np.fft.fft2(unsmoothShiftedRateMap, s=[256, 256])
        shiftedFourier = shiftedFourier / (meanFr * math.sqrt(unsmoothShiftedRateMap.shape[0] * unsmoothShiftedRateMap.shape[1]))
        shiftedFourier = np.absolute(np.fft.fftshift(shiftedFourier))
        shiftedFourier = shiftedFourier.reshape(-1)
        shiftedPowers[i] = shiftedFourier
        
    threshold = np.percentile(shiftedPowers, 50)
    return threshold

# Calculates the average ring power of a random distribution of the fourier spectrogram.
# The fourier spectrogram is either shuffled by randomly shuffling the rate map at a specified 
# chunk size or by randomly shuffling the spike train. Variable shuffleType specified as either
# 'rate map' or 'spiketrain'
def fourier_rings_significance(rateMap, spiketrain, meanFr, dt, posx, posy, area, shuffleType='rate map', chunkSize=1):
    maxRadius = math.ceil((area / (4 * math.pi)) - 1)
    shuffleAverageRingPower = np.empty((150, maxRadius))
    
    threshold = calculate_polar_threshold(meanFr, spiketrain, dt, posx, posy)
    
    for i in range(150):
        if (shuffleType == 'rate map'): 
            # Shuffle rate map pixels to determine random distribution
            unsmoothShuffleRateMap = shuffle_rate_map(rateMap, chunkSize)
            unsmoothShuffleRateMap = unsmoothShuffleRateMap - meanFr
            unsmoothShuffleRateMap[np.where(np.isnan(unsmoothShuffleRateMap))] = 0
        
        elif (shuffleType == 'spiketrain'):
        # Shuffle spiketrain to determine random distribution 
            minShift = math.ceil(20 / dt)
            maxShift = len(spiketrain) - (20 / dt)
            randShift = int(round(minShift + random.uniform(0, 1) * (maxShift - minShift)))
            
            shiftedSpiketrain = np.roll(spiketrain, randShift)
            shiftedFiringRate = np.reshape(shiftedSpiketrain / dt, (len(shiftedSpiketrain), 1))
            
            unsmoothShuffleRateMap, _ = calculate_2d_tuning_curve(posx, posy, shiftedFiringRate, 20, 0, 100)
            unsmoothShuffleRateMap = unsmoothShuffleRateMap - meanFr
            unsmoothShuffleRateMap[np.where(np.isnan(unsmoothShuffleRateMap))] = 0

        shuffleFourier = np.fft.fft2(unsmoothShuffleRateMap, s=[256, 256])
        shuffleFourier = shuffleFourier / (meanFr * math.sqrt(rateMap.shape[0] * rateMap.shape[1]))
        shuffleFourier = np.fft.fftshift(shuffleFourier)
        shuffleFourier = np.absolute(shuffleFourier)
                        
        shufflePolar = shuffleFourier - threshold
        shufflePolar[np.where(shufflePolar < 0)] = 0
                        
        averageRingPower, radii = analyze_fourier_rings(shufflePolar, area)
        averageRingPower = np.reshape(averageRingPower, (1, maxRadius))
            
        shuffleAverageRingPower[i, :] = averageRingPower

    return shuffleAverageRingPower
