import numpy as np

def find_cluster(x, xval):
    """
    Find clusters of data in an ndarray that satisfy a certain condition.


    :param x: The array containing the data for the cluster search.
    :type x: ndarray

    :param xval: The value of x that has to be satisfied for clustering.
    :type xval: integer, float


    :returns: 2-tuple

        * i0:
            The index of each cluster starting point.

        * clustersize:
            The corresponding lengths of each cluster.

    :rtype: (list, list)


    Example
    -------
        >>> x = np.int32(np.round(np.random.rand(20)+0.1))
        >>> i0, clustersize = find_cluster(x, 1)

    """
    # Cluster information list
    a = []
    # Initial (place holder) values for cluster start and end points
    kstart = -1
    kend = -1
    # Going through each value of x
    for i, xi in enumerate(x):
        if xi == xval:
            # Assigning cluster starting point
            if kstart == -1:
                kstart = i
            # Assigning cluster end point for particular case
            # when there is an xval in the last position of x
            if i == len(x)-1:
                kend = i
        else:
            # Assigning cluster end point
            if kstart != -1 and kend == -1:
                kend = i-1
        # Updating cluster information list
        # and resetting kstart and kend
        if kstart != -1 and kend != -1:
            a.append(kstart)
            a.append(kend)
            kstart = -1
            kend = -1
    # Assigning indeces of cluster starting points
    # (Every other list element starting from position 0)
    i0 = a[0:-1:2]
    # Assigning cluster sizes
    # (Every other list element starting from position 1)
    clustersize = list(np.array(a[1::2]) - np.array(i0) + 1)
    # Case where cluster size is ZERO
    if len(i0) == 0:
        i0 = []
        clustersize = []
    return i0, clustersize

def main():
  find_cluster(x,xval)
if __name__ == '__main__':
  main()