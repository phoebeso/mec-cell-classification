import numpy as np
import math
import matplotlib.pyplot as plt

def plot_canonical_scoring_method(smoothRateMap, correlationMatrix, rotations, correlations, gridScore, circularMatrix, threshold):
    displayCircularMatrix = np.copy(circularMatrix)
    displayCircularMatrix[np.where(np.isnan(displayCircularMatrix))] = -1

    plt.figure(figsize=(10,10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('Firing Rate Map')
    plt.imshow(smoothRateMap)
    plt.colorbar()
    plt.axis('off')
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('Autocorrelation Matrix')
    plt.imshow(correlationMatrix, vmin=-1, vmax=1)
    plt.colorbar()
    plt.axis('off')
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('Circular Autocorrelation Matrix')
    plt.imshow(displayCircularMatrix, vmin=-1, vmax=1)
    plt.colorbar()
    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
    plt.xlabel('Threshold: ' + str(threshold))
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('Periodicity')
    plt.plot(rotations, correlations)
    plt.xlabel('Correlation\nGrid Score: ' + str(gridScore))
    plt.ylabel('Rotation (deg)')
    plt.xlim(0, 360)
    plt.ylim(np.min(correlations), 1)
    
    plt.tight_layout()
    plt.show()
    
def plot_analyze_correlation_periodicity(collapsePartitionData, maxCollapseValues):
    for i in range(len(collapsePartitionData)):
        nPartitions = collapsePartitionData[i].get_nPartitions()
        sumPartitions = collapsePartitionData[i].get_sumPartitions()
            
        theta = np.linspace(0.0, 2 * np.pi, len(sumPartitions), endpoint=False)
        theta[np.where(sumPartitions < 0)] += np.pi
        sumPartitions = np.abs(sumPartitions)
        width = (2 * np.pi) / len(sumPartitions)
        ax = plt.subplot(111, projection='polar')
        bars = ax.bar(theta, sumPartitions, width=width)
            
        for r, bar in zip(sumPartitions, bars):
            bar.set_alpha(0.5)
            
        plt.title(str(nPartitions) + ' Collapsed Partitions')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.show()
    
    barlist = plt.bar(np.arange(8), maxCollapseValues, align='center', alpha=0.5)
    barlist[np.where(maxCollapseValues == max(maxCollapseValues))[0][0]].set_color('r')
    plt.xticks(np.arange(8), (3, 4, 5, 6, 7, 8, 9, 10))
    plt.xlabel('Number of Partitions')
    plt.ylabel('Max Collapsed Value')
    plt.show()
    
def plot_fourier_scoring(fourierSpectrogram, polarSpectrogram):
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

def plot_polar_components(polarComponents):
    for i in range(polarComponents.shape[2]):
        component = polarComponents[:, :, i]
        inverseComponent = np.real(np.fft.ifft2(np.fft.ifftshift(component)))
        
        plt.subplot(1, 3, 1)
        plt.imshow(component)
        plt.axis('off')
            
        plt.subplot(1, 3, 2)
        plt.imshow(inverseComponent)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(inverseComponent[:20, :20])
        plt.axis('off')
            
        plt.tight_layout()
        plt.show()

def plot_rho_mean_power(rhoMeanPower, localMaxima):
    theta = (np.arange(360) * math.pi) / 180

    theta2 = np.zeros(3*len(localMaxima))
    theta2[::3] = localMaxima
    theta2[1::3] = localMaxima
    theta2[2::3] = localMaxima

    rhos2 = np.zeros(3*len(localMaxima))
    rhos2[1::3] = np.ones(len(localMaxima)) * math.ceil(max(rhoMeanPower))
      
    plt.polar(theta, rhoMeanPower)
    plt.polar(theta2, rhos2)
    plt.show()

def plot_ring_power(averageRingPower, radii, maxRadius, title):
    plt.plot(radii, averageRingPower)
    plt.title('Power vs. Radius for ' + title)
    plt.xlabel('Inner Radius Length')
    plt.ylabel('Average Power')
    plt.xlim(0, maxRadius)
    plt.show()

def plot_ring_random_distribution(randomDistribution, averageRingPower, radii, confidenceInterval, shuffleRateMap, shuffleFourier, maxRadius, chunkSize):
    radii = np.arange(0, maxRadius)
    plt.plot(radii, randomDistribution)
    plt.plot(radii, averageRingPower)
    ax = plt.gca()
    ax.fill_between(radii, confidenceInterval[0], confidenceInterval[1], alpha=0.4)
    plt.title('Random distribution with rate map shuffling with chunk size = ' + str(chunkSize))
    plt.xlabel('Inner Radius Length')
    plt.ylabel('Average Power')
    plt.xlim(0, maxRadius)
    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(shuffleRateMap)
    plt.title('Shuffled Rate Map')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(shuffleFourier)
    plt.title('Shuffled Fourier Spectrogram')
    plt.axis('off')
    plt.show()
