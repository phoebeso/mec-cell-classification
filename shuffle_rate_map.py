# Shuffles rate map in groups of pixels of a given size

import numpy as np
import random
import math

def shuffle_rate_map(rateMap, chunkSize):
    numRow = rateMap.shape[0]
    numCol = rateMap.shape[1]
    flatRateMap = np.ndarray.flatten(rateMap)
    shuffleRateMap = np.empty(len(rateMap.flat))
    # If index = 1 a pixel exists there and if index = 0 a pixel does not exist there
    # Therefore, want to turn all pixels in rate map to 0 (they've all been moved)
    # and all pixels in shuffle rate map to 1 (all new locations have been filled)
    base = np.arange(0, chunkSize*math.floor(numCol/chunkSize), chunkSize)
    rateMapIndices = np.empty(0)
    for i in range(math.floor(numRow/chunkSize)):
        newIndices = base + (chunkSize * numCol * i)
        rateMapIndices = np.append(rateMapIndices, newIndices)
    shuffleIndices = np.copy(rateMapIndices)
    
    # Choose a chunk from the rate map and move it to a random chunk location 
    # in the shuffled rate map 
    numChunks = math.floor(numRow/chunkSize) * math.floor(numCol/chunkSize)
    for i in range(numChunks):
        # Finds chunk of rate map
        randRateMap = random.randint(0, len(rateMapIndices)-1)
        rateMapIndex = int(rateMapIndices[randRateMap])
        rateMapChunkIndices = get_chunk_indices(rateMapIndex, numCol, chunkSize)
        rateMapChunk = flatRateMap[rateMapChunkIndices]
        rateMapIndices = np.delete(rateMapIndices, randRateMap)
        
        # Finds chunk of indices in shuffle to insert chunk of rate map into
        randShuffle = random.randint(0, len(shuffleIndices)-1)
        shuffleIndex = int(shuffleIndices[randShuffle])
        shuffleChunkIndices = get_chunk_indices(shuffleIndex, numCol, chunkSize)
        shuffleRateMap[shuffleChunkIndices] = rateMapChunk
        shuffleIndices = np.delete(shuffleIndices, randShuffle)
     
    shuffleRateMap = np.reshape(shuffleRateMap, rateMap.shape)
    return shuffleRateMap

# rateMapSize = number of cols in rate map
# chunkSize = dimension of chunk the size, ie. 2x2 has chunk size 2 
def get_chunk_indices(index, rateMapSize, chunkSize):
    chunkIndices = np.arange(index, index+chunkSize)
    if chunkSize > 1:
        for i in range(1, chunkSize):
            newChunkIndices = np.arange(index+(i*rateMapSize), index+(i*rateMapSize)+chunkSize)
            chunkIndices = np.append(chunkIndices, newChunkIndices)
    return chunkIndices
    