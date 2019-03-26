import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

def plot_canonical_scoring_method(filename, foldername, rateMap, meanFr, correlationMatrix, rotations, correlations, gridScore, circularMatrix, threshold):
    displayCircularMatrix = np.copy(circularMatrix)
    displayCircularMatrix[np.where(np.isnan(displayCircularMatrix))] = -1
    
    plt.figure(figsize=(10, 10))

    plt.suptitle(filename + ' Canonical Scoring Method')

    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('Firing Rate Map\nMean Firing Rate: ' + str(np.round(meanFr, 5)))
    plt.imshow(rateMap)
    plt.colorbar()
    plt.axis('off')

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('Autocorrelation Matrix')
    plt.imshow(correlationMatrix, vmin=-1, vmax=1)
    plt.colorbar()
    plt.axis('off')
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('Circular Autocorrelation Matrix\nThreshold: ' + str(threshold))
    plt.imshow(displayCircularMatrix, vmin=-1, vmax=1)
    plt.colorbar()
    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('Periodicity\nGrid Score: ' + str(gridScore))
    plt.plot(rotations, correlations)
    plt.ylabel('Correlation')
    plt.xlabel('Rotation (deg)')
    plt.xlim(0, 360)
    plt.ylim(np.min(correlations), 1)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.925)

    plt.savefig(foldername + '/' + filename + '_canonical.png', dpi=300)
    plt.close()
    #plt.show()
    
def plot_analyze_correlation_periodicity(filename, foldername, collapsePartitionData, maxCollapseValues):
    plt.figure(figsize=(10, 10))

    plt.suptitle(filename + ' Correlation Partitioning Method')

    for i in range(len(collapsePartitionData)):
        nPartitions = collapsePartitionData[i].get_nPartitions()
        sumPartitions = collapsePartitionData[i].get_sumPartitions()
            
        theta = np.linspace(0.0, 2 * np.pi, len(sumPartitions), endpoint=False)
        theta[np.where(sumPartitions < 0)] += np.pi
        sumPartitions = np.abs(sumPartitions)
        width = (2 * np.pi) / len(sumPartitions)
        ax = plt.subplot(3, 4, i+1, projection='polar')    
        bars = ax.bar(theta, sumPartitions, width=width)
            
        for r, bar in zip(sumPartitions, bars):
            bar.set_alpha(0.5)
   
        ax.set_title(str(nPartitions) + ' Collapsed Partitions')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
    ax = plt.subplot(3, 1, 3)
    barlist = plt.bar(np.arange(8), maxCollapseValues, align='center', alpha=0.5)
    barlist[np.where(maxCollapseValues == max(maxCollapseValues))[0][0]].set_color('r')
    plt.xticks(np.arange(8), (3, 4, 5, 6, 7, 8, 9, 10))
    plt.xlabel('Number of Partitions')
    plt.ylabel('Max Collapsed Value')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    plt.savefig(foldername + '/' + filename + '_partitioning.png', dpi=300)
    plt.close()
    #plt.show()
    
def plot_fourier_scoring(filename, foldername, fourierSpectrogram, polarSpectrogram, maxPower, isPeriodic):
    plt.figure(figsize=(6, 4))

    plt.suptitle(filename + ' Spectrograms')

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('Fourier Spectrogram')
    plt.imshow(fourierSpectrogram)
    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
    plt.xlabel('Max Power: ' + str(maxPower) + '\nIs Periodic: ' + str(isPeriodic))
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('Polar Spectrogram')
    plt.imshow(polarSpectrogram)
    plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    plt.savefig(foldername + '/' + filename + '_spectrograms.png', dpi=300)
    plt.close()
    #plt.show()

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

def plot_ring_power(filename, averageRingPower, radii, area, maxRadius):
    plt.plot(radii, averageRingPower)
    plt.title(filename + ' Average Power vs. Radius, annulus area=' + str(area))
    plt.xlabel('Inner Radius Length')
    plt.ylabel('Average Power')
    plt.xlim(0, maxRadius)
    plt.show()

def plot_ring_random_distribution(randomDistribution, averageRingPower, radii, confidenceInterval, shuffleRateMap, shuffleFourier, maxRadius, chunkSize=None):
    radii = np.arange(0, maxRadius)
    plt.plot(radii, randomDistribution, label="Random distribution")
    plt.plot(radii, averageRingPower, label="Average power distribution")
    ax = plt.gca()
    ax.fill_between(radii, confidenceInterval[0], confidenceInterval[1], alpha=0.4)
    if chunkSize:
        plt.title('Random distribution with rate map shuffling, chunk size=' + str(chunkSize))
    else:
        plt.title('Random distribution with spiketrain shuffling')
    plt.xlabel('Inner Radius Length')
    plt.ylabel('Average Power')
    plt.xlim(0, maxRadius)
    plt.legend()
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

def plot_ring_power_distributions(filename, foldername, averageRingPower, radii, area, maxRadius, randomDistributionsData):
    plt.figure(figsize=(13, 15))

    plt.suptitle(filename + ' Power Spectrogram Annulus Analysis Method')

    ax = plt.subplot(4, 2, 1)
    ax.set_title('Average power vs. radius, annulus area=' + str(area))
    plt.plot(radii, averageRingPower, color='C0')
    plt.xlabel('Inner radius length')
    plt.ylabel('Average power')
    plt.xlim(0, maxRadius)

    randomDistributions = np.empty((len(randomDistributionsData)*150, maxRadius))
    for i in range(len(randomDistributionsData)):
        chunkSize, shuffleAverageRingPower = randomDistributionsData[i]
        randomDistribution = np.mean(shuffleAverageRingPower, axis=0)
        standardDeviation = np.std(shuffleAverageRingPower, axis=0)
        marginError = 1.96 * (standardDeviation / math.sqrt(150))
        confidenceInterval = np.empty((2, maxRadius))
        confidenceInterval[0, :] = randomDistribution - marginError
        confidenceInterval[1, :] = randomDistribution + marginError

        ax = plt.subplot(4, 2, i+2)
        ax.set_title('Random distribution with rate map shuffling, chunk size=' + str(chunkSize))
        plt.plot(radii, averageRingPower, label='Average power distribution', color='C0')
        plt.plot(radii, randomDistribution, label='Random distribution', color='C1')
        ax.fill_between(radii, confidenceInterval[0], confidenceInterval[1], color='C1', alpha=0.4)

        plt.xlabel('Inner radius length')
        plt.ylabel('Average power')
        plt.xlim(0, maxRadius)
        plt.legend()

        randomDistributions[i*150:(i+1)*150, :] = shuffleAverageRingPower

    randomDistributionAverage = np.mean(randomDistributions, axis=0)
    standardDeviation = np.std(randomDistributions, axis=0)
    marginError = 1.96 * (standardDeviation / math.sqrt(len(randomDistributions)))
    confidenceInterval = np.empty((2, maxRadius))
    confidenceInterval[0, :] = randomDistributionAverage - marginError
    confidenceInterval[1, :] = randomDistributionAverage + marginError

    ax = plt.subplot(4, 2, 7)
    ax.set_title('Average random power distribution')
    plt.plot(radii, averageRingPower, label='Average power distribution', color='C0')
    plt.plot(radii, randomDistributionAverage, label='Average random distribution', color='C1')
    ax.fill_between(radii, confidenceInterval[0], confidenceInterval[1], color='C1', alpha=0.4)
    plt.xlabel('Inner radius length')
    plt.ylabel('Average power')
    plt.xlim(0, maxRadius)
    plt.legend()

    randomDistributionAverage = np.reshape(randomDistributionAverage, (len(randomDistributionAverage), 1))
    difference = averageRingPower - randomDistributionAverage

    ax = plt.subplot(4, 2, 8)
    ax.set_title('Difference between average ring power and random distribution power')
    plt.plot(radii, difference, color='C0')
    plt.xlabel('Inner radius Length')
    plt.ylabel('Difference')
    plt.xlim(0, maxRadius)
    plt.axhline(0, color='black')

    plt.tight_layout()
    plt.subplots_adjust(top=0.925)

    plt.savefig(foldername + '/' + filename + '_ring_power_distributions.png', dpi=300)
    plt.close()
    #plt.show()

    return difference

def plot_basic_info(filename, foldername, rateMap, gridScore, meanFr, radii, difference, maxRadius):
    plt.figure(figsize=(12, 6))

    plt.suptitle(filename)

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('Rate Map\nGrid Score: ' + str(gridScore) + '\nMean Firing Rate: ' + str(np.round(meanFr, 5)))
    plt.imshow(rateMap)
    plt.colorbar()
    plt.axis('off')
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('Smoothed Difference between average ring power and random distribution power')
    plt.plot(radii, difference, color='C0')
    plt.xlabel('Inner radius Length')
    plt.ylabel('Difference')
    plt.xlim(0, maxRadius)
    plt.axhline(0, color='black')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.savefig(foldername + '/basic_info/' + filename + '_basic_info.png', dpi=300)
    plt.close()
    #plt.show()
