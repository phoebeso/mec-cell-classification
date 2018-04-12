import os
import math
import scipy.signal as signal
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from calculate_2d_tuning_curve import calculate_2d_tuning_curve
from calculate_spatial_periodicity import calculate_spatial_periodicity
from calculate_correlation_matrix import calculate_correlation_matrix
from analyze_periodicity import analyze_periodicity
from fourier_transform import fourier_transform, analyze_fourier, analyze_fourier_rings, analyze_polar_spectrogram

files = os.listdir('./SargoliniMoser2006')

boxSize = 100
nPosBins = 20

def load_data(filePath):
    # Loads data from file
    # mat_contents contains the following variables:
    # x1 = array with the x-positions for the first tracking LED
    # x2 = array with the x-positions for the second tracking LED
    # y1 = array with the y-positions for the first tracking LED
    # y2 = array with the y-positions for the second tracking LED
    # t = array with the position timestamps 
    # ts = array with the cell spike timestamps
    # filePath = './SargoliniMoser2006/' + file 
    mat_contents = sio.loadmat(filePath)
    x1 = np.array(mat_contents['x1'])
    x2 = np.array(mat_contents['x2'])
    y1 = np.array(mat_contents['y1'])
    y2 = np.array(mat_contents['y2'])
    t = np.array(mat_contents['t'])
    ts = np.array(mat_contents['ts'])

    # Adjusts position vectors so that position values range from 0-100 instead of -50-50
    x1 = x1 + 50
    x2 = x2 + 50
    y1 = y1 + 50
    y2 = y2 + 50
    posx = (x1 + x2) / 2
    posy = (y1 + y2) / 2

    # Calculates the spiketrain
    dt = t[3] - t[2]
    dt = dt[0]
    timebins = np.append(t, t[-1] + dt)
    spiketrain = np.histogram(ts, timebins)[0]

    # Smooths firing rate
    filterArray = np.reshape(signal.gaussian(9, 2), (9, 1))
    filterArray = filterArray / np.sum(filterArray)
    firingRate = spiketrain / dt
    firingRate = np.reshape(firingRate, (firingRate.shape[0], 1))
    smoothFiringRate = signal.convolve(firingRate, filterArray, 'same')

    return posx, posy, t, dt, spiketrain, smoothFiringRate

## Loops through all the files
## for file in files:
#for i in range(1):
##    if (not file.endswith('.mat')):
##        continue
#
#    # Load data from file
#    # posx, posy, t, dt, spiketrain, smoothFiringRate = load_data('./SargoliniMoser2006/11343-08120502_t8c2.mat')
#    # posx, posy, t, dt, spiketrain, smoothFiringRate = load_data('./SargoliniMoser2006/11207-21060503_t8c1.mat')
#    
#    # Calculates the unsmoothed and smoothed rate map
#    # Smoothed rate map used for correlation calculations, unsmoothed rate map used for fourier analysis
#    unsmoothRateMap, smoothRateMap = calculate_2d_tuning_curve(posx, posy, smoothFiringRate, nPosBins, 0, boxSize)
#
#    # Calculates the correlation matrix from the smooted rate map
#    correlationMatrix = calculate_correlation_matrix(smoothRateMap)
#
#    # Determines the spatial periodicity of the correlation matrix by calculating the correlation of the 
#    # matrix in intervals of 6 degrees 
#    rotations, correlations, gridScore, circularMatrix, threshold = calculate_spatial_periodicity(correlationMatrix)
#    displayCircularMatrix = np.copy(circularMatrix)
#    displayCircularMatrix[np.where(np.isnan(displayCircularMatrix))] = -1
#    
#    figure1 = plt.figure(1)
#    plt.subplot(2, 2, 1)
#    plt.title('Firing Rate Map')
#    plt.imshow(smoothRateMap)
#    plt.colorbar()
#    plt.axis('off')
#    
#    plt.subplot(2, 2, 2)
#    plt.title('Autocorrelation Matrix')
#    plt.imshow(correlationMatrix, vmin=-1, vmax=1)
#    plt.colorbar()
#    plt.axis('off')
#    
#    plt.subplot(2, 2, 3)
#    plt.title('Circular Autocorrelation Matrix')
#    plt.imshow(displayCircularMatrix, vmin=-1, vmax=1)
#    plt.colorbar()
#    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
#    plt.xlabel('Threshold: ' + str(threshold))
#    
#    plt.subplot(2, 2, 4)
#    plt.title('Periodicity')
#    plt.plot(rotations, correlations)
#    plt.xlabel('Correlation\nGrid Score: ' + str(gridScore))
#    plt.ylabel('Rotation (deg)')
#    plt.xlim(0, 360)
#    plt.ylim(np.min(correlations), 1)
#    
#    plt.tight_layout()
#    plt.show()
#    
#    # Partitions the correlation periodicity curve into 3-10 periods and 
#    # collapses/sums the partitioned data 
#    collapsePartitionData, maxCollapseValues = analyze_periodicity(rotations, correlations)
#    
#    for i in range(len(collapsePartitionData)):
#        nPartitions = collapsePartitionData[i].get_nPartitions()
#        sumPartitions = collapsePartitionData[i].get_sumPartitions()
#        
#        theta = np.linspace(0.0, 2 * np.pi, len(sumPartitions), endpoint=False)
#        theta[np.where(sumPartitions < 0)] += np.pi
#        sumPartitions = np.abs(sumPartitions)
#        width = (2 * np.pi) / len(sumPartitions)
#        ax = plt.subplot(111, projection='polar')
#        bars = ax.bar(theta, sumPartitions, width=width)
#        
#        for r, bar in zip(sumPartitions, bars):
#            bar.set_alpha(0.5)
#        
#        plt.title(str(nPartitions) + ' Collapsed Partitions')
#        ax.set_yticklabels([])
#        ax.set_xticklabels([])
#        plt.show()
#        
#    barlist = plt.bar(np.arange(8), maxCollapseValues, align='center', alpha=0.5)
#    barlist[np.where(maxCollapseValues == max(maxCollapseValues))[0][0]].set_color('r')
#    plt.xticks(np.arange(8), (3, 4, 5, 6, 7, 8, 9, 10))
#    plt.xlabel('Number of Partitions')
#    plt.ylabel('Max Collapsed Value')
#    plt.show()
#
#    # Calculates thsae two-dimensional Fourier spectrogram
#    adjustedRateMap = unsmoothRateMap - np.nanmean(unsmoothRateMap)
#    meanFr = np.sum(spiketrain) / t[-1]
#    meanFr = meanFr[0]
#    maxRate = np.max(unsmoothRateMap)
#    maxAdjustedRate = np.max(adjustedRateMap)
#    fourierSpectrogram, polarSpectrogram, beforeMaxPower, maxPower, isPeriodic = fourier_transform(adjustedRateMap, meanFr, spiketrain, dt, posx, posy)
#    
#    figure2 = plt.figure(2)
#    plt.subplot(1, 2, 1)
#    plt.title('Fourier Spectrogram')
#    plt.imshow(fourierSpectrogram)
#    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
#    
#    plt.subplot(1, 2, 2)
#    plt.title('Polar Spectrogram')
#    plt.imshow(polarSpectrogram)
#    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
#
#    plt.show()
#    
#    polarComponents = analyze_fourier(fourierSpectrogram, polarSpectrogram)
#    
#    for i in range(polarComponents.shape[2]):
#        component = polarComponents[:, :, i]
#        inverseComponent = np.real(np.fft.ifft2(np.fft.ifftshift(component)))
#        
#        plt.subplot(1, 3, 1)
#        plt.imshow(component)
#        
#        plt.subplot(1, 3, 2)
#        plt.imshow(inverseComponent)
#        
#        plt.subplot(1, 3, 3)
#        plt.imshow(inverseComponent[:20, :20])
#        
#        plt.tight_layout()
#        plt.show()
#    
#    averageRingPower, radii = analyze_fourier_rings(fourierSpectrogram, 1605)
#    
#    plt.plot(radii, averageRingPower)
#    plt.xlabel('Inner Radius Length')
#    plt.ylabel('Average Power')
#    plt.show()
#    
#    rhoMeanPower, localMaxima = analyze_polar_spectrogram(polarSpectrogram)
#    theta = (np.arange(360) * math.pi) / 180
#    
#    plt.polar(theta, rhoMeanPower)
#    plt.show()
#    
#    rhoMeanPower, localMaxima = analyze_polar_spectrogram(polarSpectrogram)
#        
#    theta = (np.arange(360) * math.pi) / 180
#    
#    theta2 = np.zeros(3*len(localMaxima))
#    theta2[::3] = localMaxima
#    theta2[1::3] = localMaxima
#    theta2[2::3] = localMaxima
#    
#    rhos2 = np.zeros(3*len(localMaxima))
#    rhos2[1::3] = np.ones(len(localMaxima)) * math.ceil(max(rhoMeanPower))
#      
#    plt.polar(theta, rhoMeanPower)
#    plt.polar(theta2, rhos2)
#    plt.show()
    
def combine_rate_maps():
    nPosBins = 20
    boxSize = 100
    
    posx1, posy1, t1, dt1, spiketrain1, smoothFiringRate1 = load_data('./SargoliniMoser2006/11343-08120502_t8c2.mat')
    posx2, posy2, t2, dt2, spiketrain2, smoothFiringRate2 = load_data('./SargoliniMoser2006/11207-21060503_t8c1.mat')
    
    unsmoothRateMap1, smoothRateMap1 = calculate_2d_tuning_curve(posx1, posy1, smoothFiringRate1, nPosBins, 0, boxSize)
    spiketrain1 = np.reshape(spiketrain1, (len(spiketrain1), 1))
    unsmoothRateMap2, smoothRateMap2 = calculate_2d_tuning_curve(posx2, posy2, smoothFiringRate2, nPosBins, 0, boxSize)
    spiketrain2 = np.reshape(spiketrain2, (len(spiketrain2), 1))
    
    combineUnsmoothRateMap = unsmoothRateMap1 + unsmoothRateMap2
    combineSmoothRateMap = smoothRateMap1 + smoothRateMap2
    combineSpiketrain = np.vstack((spiketrain1, spiketrain2))
    combinePosx = np.vstack((posx1, posx2))
    combinePosy = np.vstack((posy1, posy2))
    
    correlationMatrix = calculate_correlation_matrix(combineSmoothRateMap)
    
    rotations, correlations, gridScore, circularMatrix, threshold = calculate_spatial_periodicity(correlationMatrix)
    displayCircularMatrix = np.copy(circularMatrix)
    displayCircularMatrix[np.where(np.isnan(displayCircularMatrix))] = -1
    
    figure1 = plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.title('Firing Rate Map')
    plt.imshow(combineSmoothRateMap)
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title('Autocorrelation Matrix')
    plt.imshow(correlationMatrix, vmin=-1, vmax=1)
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title('Circular Autocorrelation Matrix')
    plt.imshow(displayCircularMatrix, vmin=-1, vmax=1)
    plt.colorbar()
    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
    plt.xlabel('Threshold: ' + str(threshold))
    
    plt.subplot(2, 2, 4)
    plt.title('Periodicity')
    plt.plot(rotations, correlations)
    plt.xlabel('Correlation\nGrid Score: ' + str(gridScore))
    plt.ylabel('Rotation (deg)')
    plt.xlim(0, 360)
    plt.ylim(np.min(correlations), 1)
    
    plt.tight_layout()
    plt.show()

    # Fourier analysis for combined cells 
    adjustedRateMap = combineUnsmoothRateMap - np.nanmean(combineUnsmoothRateMap)
    meanFr = np.sum(combineSpiketrain) / (t1[-1] + t2[-1])
    meanFr = meanFr[0]
    fourierSpectrogram, polarSpectrogram, beforeMaxPower, maxPower, isPeriodic = fourier_transform(adjustedRateMap, meanFr, combineSpiketrain, dt1, combinePosx, combinePosy)
    
    # Fourier analysis for cell 1
    adjustedRateMap1 = unsmoothRateMap1 - np.nanmean(unsmoothRateMap1)
    meanFr1 = np.sum(spiketrain1) / t1[-1]
    maxAdjustedRate1 = np.max(adjustedRateMap1)
    fourierSpectrogram1, polarSpectrogram1, beforeMaxPower1, maxPower1, isPeriodic1 = fourier_transform(adjustedRateMap1, meanFr1, spiketrain1, dt1, posx1, posy1)
    
    # Fourier analysis for cell 2
    adjustedRateMap2 = unsmoothRateMap2 - np.nanmean(unsmoothRateMap2)
    meanFr2 = np.sum(spiketrain2) / t2[-1]
    maxAdjustedRate2 = np.max(adjustedRateMap2)
    fourierSpectrogram2, polarSpectrogram2, beforeMaxPower2, maxPower2, isPeriodic2 = fourier_transform(adjustedRateMap2, meanFr2, spiketrain2, dt2, posx2, posy2)

    figure2 = plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.title('Fourier Spectrogram')
    plt.imshow(fourierSpectrogram)
    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
    
    plt.subplot(1, 2, 2)
    plt.title('Polar Spectrogram')
    plt.imshow(polarSpectrogram)
    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')

    plt.show()
    
    combineFourierSpectrogram = fourierSpectrogram1 + fourierSpectrogram2
    combinePolarSpectrogram = polarSpectrogram1 + polarSpectrogram2
    
    figure3 = plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.title('Combine Fourier Spectrogram')
    plt.imshow(combineFourierSpectrogram)
    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
    
    plt.subplot(1, 2, 2)
    plt.title('Combine Polar Spectrogram')
    plt.imshow(combinePolarSpectrogram)
    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
    
    plt.show()
    
    polarComponents = analyze_fourier(fourierSpectrogram, polarSpectrogram)
    
    for i in range(polarComponents.shape[2]):
        component = polarComponents[:, :, i]
        inverseComponent = np.real(np.fft.ifft2(np.fft.ifftshift(component)))
        
        plt.subplot(1, 3, 1)
        plt.imshow(component)
        
        plt.subplot(1, 3, 2)
        plt.imshow(inverseComponent)
        
        plt.subplot(1, 3, 3)
        plt.imshow(inverseComponent[:20, :20])
        
        plt.tight_layout()
        plt.show()
        
    averageRingPower, radii = analyze_fourier_rings(fourierSpectrogram, 1605)
   
    plt.plot(radii, averageRingPower)
    plt.xlabel('Inner Radius Length')
    plt.ylabel('Average Power')
    plt.show()
    
    averageRingPower, radii = analyze_fourier_rings(fourierSpectrogram1, 1605)
   
    plt.plot(radii, averageRingPower)
    plt.xlabel('Inner Radius Length')
    plt.ylabel('Average Power')
    plt.show()
    
    averageRingPower, radii = analyze_fourier_rings(fourierSpectrogram2, 1605)
   
    plt.plot(radii, averageRingPower)
    plt.xlabel('Inner Radius Length')
    plt.ylabel('Average Power')
    plt.show()
    
    
# combine_rate_maps()