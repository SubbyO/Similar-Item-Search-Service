# Code adapted from https://github.com/scipy/scipy/blob/v1.5.2/scipy/spatial/kdtree.py

import numpy as np
from heapq import heappush, heappop
import sys


def minkowski_distance_p(x, y, p=2):
    """
    Compute the pth power of the L**p distance between two arrays.
    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.
    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Examples
    --------
    >>> from scipy.spatial import minkowski_distance_p
    >>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
    array([2, 1])
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Find smallest common datatype with float64 (return type of this function) - addresses #10262.
    # Don't just cast to float64 for complex input case.
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype), 'float64')

    # Make sure x and y are NumPy arrays of correct datatype.
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)

    if p == np.inf:
        return np.amax(np.abs(y-x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y-x), axis=-1)
    else:
        return np.sum(np.abs(y-x)**p, axis=-1)


def minkowski_distance(x, y, p=2):
    """
    Compute the L**p distance between two arrays.
    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Examples
    --------
    >>> from scipy.spatial import minkowski_distance
    >>> minkowski_distance([[0,0],[0,0]], [[1,1],[0,1]])
    array([ 1.41421356,  1.        ])
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p)**(1./p)


class KDTree(object):
    """
    kd-tree for quick nearest-neighbor lookup
    This class provides an index into a set of k-D points which
    can be used to rapidly look up the nearest neighbors of any point.
    Parameters
    ----------
    data : (N,K) array_like
        The data points to be indexed. This array is not copied, and
        so modifying this data will result in bogus results.
    leafsize : int, optional
        The number of points at which the algorithm switches over to
        brute-force.  Has to be positive.
    """
    def __init__(self, data, leafsize=10):
        sys.setrecursionlimit(10000)
        self.data = data
        if self.data.dtype.kind == 'c':
            raise TypeError("KDTree does not work with complex data")

        self.n, self.m = self.data.shape
        self.leafsize = int(leafsize)
        if self.leafsize < 1:
            raise ValueError("leafsize must be at least 1")
        self.maxes = self.data.max(axis=0)
        self.mins = self.data.min(axis=0)

        self.tree = self.__build(np.arange(self.n), self.maxes, self.mins)

    def serialize(self):
        return {"n": self.n, "m": self.m, "leafsize": self.leafsize, "maxes": self.maxes, "mins": self.mins, "tree": self.tree}

    @classmethod
    def deserialize(cls, serialized, data):
        kdtree = cls.__new__(cls)
        super(KDTree, kdtree).__init__()
        kdtree.n = serialized["n"]
        kdtree.m = serialized["m"]
        kdtree.leafsize = serialized["leafsize"]
        kdtree.maxes = serialized["maxes"]
        kdtree.mins = serialized["mins"]
        kdtree.tree = serialized["tree"]
        kdtree.data = data
        return kdtree

    class node(object):
        def __lt__(self, other):
            return id(self) < id(other)

        def __gt__(self, other):
            return id(self) > id(other)

        def __le__(self, other):
            return id(self) <= id(other)

        def __ge__(self, other):
            return id(self) >= id(other)

        def __eq__(self, other):
            return id(self) == id(other)

    class leafnode(node):
        def __init__(self, idx):
            self.idx = idx
            self.children = len(idx)

    class innernode(node):
        def __init__(self, split_dim, split, less, greater):
            self.split_dim = split_dim
            self.split = split
            self.less = less
            self.greater = greater
            self.children = less.children+greater.children

    def __build(self, idx, maxes, mins):
        if len(idx) <= self.leafsize:
            return KDTree.leafnode(idx)
        else:
            d = np.argmax(maxes-mins)
            maxval = maxes[d]
            minval = mins[d]
            if maxval == minval:
                # all points are identical; warn user?
                return KDTree.leafnode(idx)
            data = self.data[idx,d].reshape((len(idx),1))

            # sliding midpoint rule; see Maneewongvatana and Mount 1999
            # for arguments that this is a good idea.
            split = (maxval+minval)/2
            less_idx = np.nonzero(data <= split)[0]
            greater_idx = np.nonzero(data > split)[0]
            if len(less_idx) == 0:
                split = np.amin(data)
                less_idx = np.nonzero(data <= split)[0]
                greater_idx = np.nonzero(data > split)[0]
            if len(greater_idx) == 0:
                split = np.amax(data)
                less_idx = np.nonzero(data < split)[0]
                greater_idx = np.nonzero(data >= split)[0]
            if len(less_idx) == 0:
                # _still_ zero? all must have the same value
                if not np.all(data == data[0]):
                    raise ValueError("Troublesome data array: %s" % data)
                split = data[0]
                less_idx = np.arange(len(data)-1)
                greater_idx = np.array([len(data)-1])

            lessmaxes = np.copy(maxes)
            lessmaxes[d] = split
            greatermins = np.copy(mins)
            greatermins[d] = split
            return KDTree.innernode(d, split,
                    self.__build(idx[less_idx],lessmaxes,mins),
                    self.__build(idx[greater_idx],maxes,greatermins))

    def __query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):

        side_distances = np.maximum(0,np.maximum(x-self.maxes,self.mins-x))
        if p != np.inf:
            side_distances **= p
            min_distance = np.sum(side_distances)
        else:
            min_distance = np.amax(side_distances)

        # priority queue for chasing nodes
        # entries are:
        #  minimum distance between the cell and the target
        #  distances between the nearest side of the cell and the target
        #  the head node of the cell
        q = [(min_distance,
              tuple(side_distances),
              self.tree)]
        # priority queue for the nearest neighbors
        # furthest known neighbor first
        # entries are (-distance**p, i)
        neighbors = []

        if eps == 0:
            epsfac = 1
        elif p == np.inf:
            epsfac = 1/(1+eps)
        else:
            epsfac = 1/(1+eps)**p

        if p != np.inf and distance_upper_bound != np.inf:
            distance_upper_bound = distance_upper_bound**p

        while q:
            min_distance, side_distances, node = heappop(q)
            if isinstance(node, KDTree.leafnode):
                # brute-force
                data = self.data[node.idx]
                ds = minkowski_distance_p(data,x[np.newaxis,:],p)
                for i in range(len(ds)):
                    if ds[i] < distance_upper_bound:
                        if len(neighbors) == k:
                            heappop(neighbors)
                        heappush(neighbors, (-ds[i], node.idx[i]))
                        if len(neighbors) == k:
                            distance_upper_bound = -neighbors[0][0]
            else:
                # we don't push cells that are too far onto the queue at all,
                # but since the distance_upper_bound decreases, we might get
                # here even if the cell's too far
                if min_distance > distance_upper_bound*epsfac:
                    # since this is the nearest cell, we're done, bail out
                    break
                # compute minimum distances to the children and push them on
                if x[node.split_dim] < node.split:
                    near, far = node.less, node.greater
                else:
                    near, far = node.greater, node.less

                # near child is at the same distance as the current node
                heappush(q,(min_distance, side_distances, near))

                # far child is further by an amount depending only
                # on the split value
                sd = list(side_distances)
                if p == np.inf:
                    min_distance = max(min_distance, abs(node.split-x[node.split_dim]))
                elif p == 1:
                    sd[node.split_dim] = np.abs(node.split-x[node.split_dim])
                    min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]
                else:
                    sd[node.split_dim] = np.abs(node.split-x[node.split_dim])**p
                    min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]

                # far child might be too far, if so, don't bother pushing it
                if min_distance <= distance_upper_bound*epsfac:
                    heappush(q,(min_distance, tuple(sd), far))

        if p == np.inf:
            return sorted([(-d,i) for (d,i) in neighbors])
        else:
            return sorted([((-d)**(1./p),i) for (d,i) in neighbors])

    def query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        """
        Query the kd-tree for nearest neighbors
        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : int, optional
            The number of nearest neighbors to return.
        eps : nonnegative float, optional
            Return approximate nearest neighbors; the kth returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real kth nearest neighbor.
        p : float, 1<=p<=infinity, optional
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
        distance_upper_bound : nonnegative float, optional
            Return only neighbors within this distance. This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.
        Returns
        -------
        d : float or array of floats
            The distances to the nearest neighbors.
            If x has shape tuple+(self.m,), then d has shape tuple if
            k is one, or tuple+(k,) if k is larger than one. Missing
            neighbors (e.g. when k > n or distance_upper_bound is
            given) are indicated with infinite distances.  If k is None,
            then d is an object array of shape tuple, containing lists
            of distances. In either case the hits are sorted by distance
            (nearest first).
        i : integer or array of integers
            The locations of the neighbors in self.data. i is the same
            shape as d.
        """
        x = np.asarray(x)
        if x.dtype.kind == 'c':
            raise TypeError("KDTree does not work with complex data")
        if np.shape(x)[-1] != self.m:
            raise ValueError("x must consist of vectors of length %d but has shape %s" % (self.m, np.shape(x)))
        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")
        retshape = np.shape(x)[:-1]
        if retshape != ():
            if k is None:
                dd = np.empty(retshape,dtype=object)
                ii = np.empty(retshape,dtype=object)
            elif k > 1:
                dd = np.empty(retshape+(k,),dtype=float)
                dd.fill(np.inf)
                ii = np.empty(retshape+(k,),dtype=int)
                ii.fill(self.n)
            elif k == 1:
                dd = np.empty(retshape,dtype=float)
                dd.fill(np.inf)
                ii = np.empty(retshape,dtype=int)
                ii.fill(self.n)
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
            for c in np.ndindex(retshape):
                hits = self.__query(x[c], k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)
                if k is None:
                    dd[c] = [d for (d,i) in hits]
                    ii[c] = [i for (d,i) in hits]
                elif k > 1:
                    for j in range(len(hits)):
                        dd[c+(j,)], ii[c+(j,)] = hits[j]
                elif k == 1:
                    if len(hits) > 0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self.__query(x, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)
            if k is None:
                return [d for (d,i) in hits], [i for (d,i) in hits]
            elif k == 1:
                if len(hits) > 0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k > 1:
                dd = np.empty(k,dtype=float)
                dd.fill(np.inf)
                ii = np.empty(k,dtype=int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
