import os, time
import numpy as np


def buildIndex(q, out, p_i, datadir):
    """
    Handle requests for one process to build search index.
    This function will be ran independently by multiple processes,
    whose outputs will be aggregated to form the index.
    Parameters
    ----------
    q : multiprocessing.Manager.Queue
        Process safe queue, which input blocks are read and distributed across processes from
    out : multiprocessing.Manager.list
        Process safe list which outputs from processes will be stored in
    p_i : int
        Id/index of process running this function
    datadir: string
        Directory where data is stored for the index
    """
    embeddingsByCategory = {}
    itemsByCategory = {}
    itemPositions = {}
    while True:
        # Process inputs from the queue until last request (None) is reached
        block, lastPosition = q.get(block=True)
        if block is None:
            embeddingShapes = {}
            for category, (_, catMap) in embeddingsByCategory.items():
                embeddingShapes[category] = catMap.shape
                del catMap
            out.append((p_i, embeddingShapes, itemsByCategory, itemPositions))
            time.sleep(1)
            return

        for i, item in enumerate(block):
            itemCategory = item['leaf_categ_id']
            itemId = item['itemId']
            embeddingsByCategory.setdefault(itemCategory, [0, None])
            itemsByCategory.setdefault(itemCategory, [])
            if itemId not in itemPositions:
                catEmbeddings = embeddingsByCategory[itemCategory]
                present = catEmbeddings[1] is not None
                if present:
                    del catEmbeddings[1]
                    catEmbeddings.append(None)
                catFile = os.path.join(datadir, str(itemCategory) + '_data_' + str(p_i) + '.array')
                catEmbeddings[1] = np.memmap(catFile, dtype='float32', mode='r+' if present else 'w+',\
                    shape=(catEmbeddings[0]+1, len(item['embedding'])))
                catEmbeddings[1][-1,:] = item['embedding']
                catEmbeddings[0] += 1
                itemsByCategory[itemCategory].append(itemId)
                itemPositions[itemId] = (lastPosition, i)
