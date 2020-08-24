from fastavro import block_reader
from tqdm import tqdm
from KDTree import KDTree
import numpy as np
import os, pickle
import sys
from multiprocessing import Manager
import multiprocessing as mp
import itertools
import util
import logging

sys.setrecursionlimit(10000)
BLOCK_LEN = 13

class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class SearchIndex(object):
    def __init__(self, datafile, datadir='', n_records=None, n_jobs=1):
        self.datadir = datadir
        self.datafile = datafile
        self.fo = open(os.path.join(self.datadir, self.datafile), "rb")
        self.reader = block_reader(self.fo)
        # Dictionary mapping categories to ordered lists of ids of all items in the category
        self.itemsByCategory = {}
        # Dictionary mapping item ids to their position in the input file
        self.itemPositions = {}
        self.logger = logging.getLogger('similar_item_service.search.SearchIndex')
        self._build_(n_records=n_records, n_jobs=n_jobs)
        # Keep track of current block and index in datafile to avoid unecessarily reseeking/iteration
        self.curBlock = None
        self.curIdx = None
        

    @classmethod
    def load(cls, fname):
        '''
        Load a pickled search index from file fname
        '''
        index = cls.__new__(cls)
        super(SearchIndex, index).__init__()
        index.logger = logging.getLogger('similar_item_service.search.SearchIndex')
        with open(fname, 'rb') as f:
            index.datadir, index.datafile, index.itemsByCategory, index.itemPositions = pickle.load(f)
        index.logger.info(f"Loading search index for {os.path.join(index.datadir, index.datafile)} from {fname}")
        index.fo = open(os.path.join(index.datadir, index.datafile), "rb")
        index.reader = block_reader(index.fo)
        index.curBlock = None
        index.curIdx = None
        return index

    def _build_(self, n_records=None, n_jobs=1):
        '''
        Build a search index using n_jobs cpu cores. If n_jobs is 0, all cores are used.
        '''
        n_jobs = mp.cpu_count() if n_jobs == 0 else max(1, n_jobs)
        self.logger.info(f"Building search index for {os.path.join(self.datadir, self.datafile)} using {n_jobs} CPU core(s)")
        if n_jobs == 1:
            lastPosition = self.fo.tell()
            nextPosition = None
            embeddingsByCategory = {}
            with tqdm(total=n_records, desc='Building search index') as pbar:
                for block in self.reader:
                    if nextPosition is not None:
                        lastPosition = nextPosition
                    nextPosition = self.fo.tell()
                    for i, item in enumerate(block):
                        itemCategory = item['leaf_categ_id']
                        itemId = item['itemId']
                        embeddingsByCategory.setdefault(itemCategory, [0, None])
                        self.itemsByCategory.setdefault(itemCategory, [])
                        if itemId not in self.itemPositions:
                            catEmbeddings = embeddingsByCategory[itemCategory]
                            present = catEmbeddings[1] is not None
                            if present:
                                del catEmbeddings[1]
                                catEmbeddings.append(None)
                            catFile = os.path.join(self.datadir, str(itemCategory) + '_data.array')
                            # Sotre and access item embedding matrix on disk to avoid overflowing system memory
                            catEmbeddings[1] = np.memmap(catFile, dtype='float32', mode='r+' if present else 'w+', shape=(catEmbeddings[0]+1, len(item['embedding'])))
                            catEmbeddings[1][-1,:] = item['embedding']
                            catEmbeddings[0] += 1
                            self.itemsByCategory[itemCategory].append(itemId)
                            self.itemPositions[itemId] = (lastPosition, i)
                        pbar.update(1)
            for _, (_, catMap) in embeddingsByCategory.items():
                del catMap
            del embeddingsByCategory
        else:
            self._buildParallel_(n_jobs, n_records=n_records)

    def _buildParallel_(self, n_jobs, n_records=None):
        manager = Manager()
        results = manager.list()
        work = manager.Queue(n_jobs)
        pool = []
        for i in range(n_jobs):
            p = Process(target=util.buildIndex, args=(work, results, i, self.datadir))
            p.start()
            pool.append(p)
        
        lastPosition = self.fo.tell()
        nextPosition = None
        iters = itertools.chain(self.reader, (None,)*n_jobs)
        with tqdm(total=n_records, desc='Building search index') as pbar:
            for block in iters:
                if nextPosition is not None:
                    lastPosition = nextPosition
                nextPosition = self.fo.tell()
                work.put((block, lastPosition))
                pbar.update(BLOCK_LEN)
        
        for p in pool:
            p.join()
            exc = p.exception
            if exc:
                raise exc
        
        self.itemPositions = {}
        self.itemsByCategory = {}
        embeddingsByCategory = {}
        self.logger.info("Joining indices")
        for p_i, embeddingShapes, items, positions in results:
            for category, itemIds in items.items():
                slc = None if category not in self.itemsByCategory else []
                self.itemsByCategory.setdefault(category, [])
                for i, itemId in enumerate(itemIds):
                    if slc is None or itemId not in self.itemPositions:
                        self.itemsByCategory[category].append(itemId)
                        if slc is not None:
                            slc.append(i)
                catFile = os.path.join(self.datadir, str(category) + '_data_' + str(p_i) + '.array')
                catEmbedding = np.memmap(catFile, dtype='float32', mode='r', shape=embeddingShapes[category])
                if slc is None:
                    embeddingsByCategory[category] = [catFile, catEmbedding]
                else:
                    catEmbeddings = embeddingsByCategory[category]
                    newShape = (catEmbeddings[1].shape[0]+len(slc), catEmbeddings[1].shape[1])
                    catEmbeddings[1] = np.memmap(catEmbeddings[0], dtype='float32', mode='r+', shape=newShape)
                    catEmbeddings[1][-len(slc):,:] = catEmbedding[slc,:]
                    del catEmbedding
                    os.remove(catFile)
            for itemId, position in positions.items():
                if itemId not in self.itemPositions:
                    self.itemPositions[itemId] = position
        for category in self.itemsByCategory.keys():
            catEmbeddings = embeddingsByCategory[category]
            shp = catEmbeddings[1].shape
            del catEmbeddings[1]
            catFile = os.path.join(self.datadir, str(category) + '_data.array')
            os.rename(catEmbeddings[0], catFile)
        del embeddingsByCategory

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.logger.info("Closing search index")
        self.fo.close()

    def _get_(self, blockStart, index):
        if self.curBlock is None or self.curBlock[0] != blockStart or self.curIdx > index:
            self.fo.seek(blockStart)
            try:
                self.curBlock = (blockStart, next(self.reader))
            except StopIteration:
                self.fo.seek(0)
                self.reader = block_reader(self.fo)
                self.fo.seek(blockStart)
                self.curBlock = (blockStart, next(self.reader))
            self.curIdx = 0
        for item in self.curBlock[1]:
            self.curIdx += 1
            if self.curIdx > index:
                return item

    def __getitem__(self, itemId):
        '''
        Return the record for a given item from the avro datafile
        '''
        self.logger.debug(f"Reading info for item {itemId}")
        if itemId not in self.itemPositions:
            self.logger.error(f"Item {itemId} not found in index")
            raise KeyError
        return self._get_(*self.itemPositions[itemId])

    def save(self, fname):
        '''
        Save index to be loaded in future runs
        '''
        self.logger.info(f"Saving search index for {os.path.join(self.datadir, self.datafile)} to {fname}")
        data = (self.datadir, self.datafile, self.itemsByCategory, self.itemPositions)
        with open(fname, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


class SearchComponent(object):
    def __init__(self, index, datadir='', n_jobs=0):
        self.index = index
        self.datadir = datadir
        self.cache = False
        # Store a kd tree for each category of items
        self.trees = {}
        self.logger = logging.getLogger('similar_item_service.search.SearchComponent')
        self._build_(n_jobs=n_jobs)
    
    @classmethod
    def load(cls, fname, index):
        '''
        Load a pickled search component from file fname, using search index index
        '''
        comp = cls.__new__(cls)
        super(SearchComponent, comp).__init__()
        comp.logger = logging.getLogger('similar_item_service.search.SearchComponent')
        comp.logger.info(f"Loading search component from {fname}")
        comp.index = index
        with open(fname, 'rb') as f:
            comp.datadir, savedtrees = pickle.load(f)
        comp.cache = True
        comp.trees = {}
        for category, savedtree in savedtrees.items():
            catFile = os.path.join(comp.datadir, str(category) + '_data.array')
            if not os.path.exists(catFile):
                comp.logger.error(f"Missing data file required for loading search component: {catFile}")
                raise FileNotFoundError
            catEmbeddings = np.memmap(catFile, dtype='float32', mode='r', shape=(savedtree["n"], savedtree["m"]))
            comp.trees[category] = KDTree.deserialize(savedtree, catEmbeddings)
        return comp
    
    def _build_(self, n_jobs=0):
        '''
        Build a search component using n_jobs cpu cores. If n_jobs is 0, all cores are used.
        '''
        n_jobs = mp.cpu_count() if n_jobs == 0 else max(1, n_jobs)
        self.logger.info(f"Building search component with {n_jobs} CPU cores")
        catEmbeddings = {}
        for category, items in self.index.itemsByCategory.items():
            catFile = os.path.join(self.datadir, str(category) + "_data.array")
            catEmbeddings[category] = np.memmap(catFile, dtype='float32', mode='r')
            catEmbeddings[category].shape = (len(items), -1)
        if n_jobs == 1:
            for i, (category, embeddings) in enumerate(catEmbeddings.items(), 1):
                self.logger.info(f"Building tree for category {category}, {i} of {len(catEmbeddings)}")
                self.trees[category] = KDTree(embeddings)
        else:
            with mp.Pool(processes=n_jobs) as pool:
                results = pool.map(KDTree, catEmbeddings.values())
            self.trees = dict(zip(catEmbeddings.keys(), results))
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.logger.info("Closing search component")
        if not self.cache:
            for category, tree in self.trees.items():
                catFile = os.path.join(self.datadir, str(category) + '_data.array')
                del tree.data
                if os.path.exists(catFile):
                    self.logger.info(f"Deleting data file for search component: {catFile}")
                    os.remove(catFile)
    
    def _similarity_(self, dist, n_features):
        '''
        Convert euclidean distance to siilarity
        '''
        return np.exp(-(dist*np.log(dist))/n_features) if dist > 0 else 1
    
    def query(self, itemId, category=None, numResults=None):
        '''
        Find the numResults most similar items to a given item (itemId) from a given category (category)
        Returns results sorted from most to least similar. If numResults is None, all items are included
        '''
        self.logger.info(f"Finding {str(numResults)+' ' if numResults is not None else ''}most similar items to item {itemId}")
        item = self.index[itemId]
        category = category if category is not None else item['leaf_categ_id']
        dists, itemIdxs = self.trees[category].query(item['embedding'], k=numResults+1 if numResults is not None else None)
        seenItem = False
        res = []
        for idx, dist in zip(itemIdxs, dists):
            itId = self.index.itemsByCategory[category][idx]
            if itId == itemId:
                seenItem = True
                i = item
            else:
                i = self.index[itId]
            res.append((itId, self._similarity_(dist, len(i['embedding'])), i['gallery_url']))
        if not seenItem:
            res = [(itemId, 1, item['gallery_url'])] + res
        return res

    def getNRandom(self, n, category=None):
        '''
        Get n random item ids from the index
        '''
        choices = list(self.index.itemPositions.keys()) if category is None else self.index.itemsByCategory[category]
        itemIds = np.random.choice(choices, n, replace=False)
        items = []
        for itemId in itemIds:
            item = self.index[itemId]
            items.append({"itemId": item["itemId"], "category": item["leaf_categ_id"], "imageUrl": item["gallery_url"]})
        return items
    
    def save(self, fname):
        '''
        Save search component to be loaded in future runs
        '''
        self.logger.info(f"Saving search component to {fname}")
        savedtrees = {category: tree.serialize() for category, tree in self.trees.items()}
        data = (self.datadir, savedtrees)
        with open(fname, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        self.cache = True
