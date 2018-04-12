import numpy as np
import matplotlib.pyplot as plt

def plot_canonical_scoring_method(smoothRateMap, correlationMatrix, rotations, correlations, gridScore, circularMatrix, threshold):
    displayCircularMatrix = np.copy(circularMatrix)
    displayCircularMatrix[np.where(np.isnan(displayCircularMatrix))] = -1
    
    figure1 = plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.title('Firing Rate Map')
    plt.imshow(smoothRateMap)
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
