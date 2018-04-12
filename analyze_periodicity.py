# Partitions the sinusoidal correlation data into n even partitions. Collapses the
# partitions by calculating the summation of all n partitions. Data is used
# for periodicity analysis.

import numpy as np

def analyze_periodicity(rotations, correlations):
    rotations = rotations[:60]
    correlations = correlations[:60]
    
    data = np.empty(8, dtype=Partition_Data)
    maxValues = np.empty(8)
    
    # Partitions the sinusoidal correlation data in 3-10 even partitions and
    # collapses the data
    for nPartitions in range(3, 11):
        partitions = np.array_split(correlations, nPartitions)
        
        # Crops all partitions to be the same length (60 // nPartitions)
        if (len(correlations) % nPartitions != 0):
            for i in range(len(partitions)):
                if (len(partitions[i]) > len(correlations) // nPartitions):
                    partitions[i] = partitions[i][:len(correlations) // nPartitions]
        
        partitionsMatrix = np.hstack(partitions)
        sumPartitions = np.sum(partitionsMatrix, axis=1)
        
        partitionData = Partition_Data(nPartitions, sumPartitions)
        
        data[nPartitions-3] = partitionData
        maxValues[nPartitions-3] = max(sumPartitions)
        
        
    return data, maxValues

class Partition_Data:
    
    def __init__(self, nPartitions, sumPartitions):
        self.nPartitions = nPartitions
        self.sumPartitions = sumPartitions
        
    def get_nPartitions(self):
        return self.nPartitions
    
    def get_sumPartitions(self):
        return self.sumPartitions
    