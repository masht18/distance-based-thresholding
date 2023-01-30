import numpy as np

def distance_thresholding(A, dist, hemiid, nbins):
    '''
    Input:
    - A: n x n x s 
        connection strengths between n edges for s subjects.
    - dist: n x n 
        edge distances
    - hemiid: n x 1 
        matrix indicating the hemisphere of each region in the graph (1 or 2)
    - n_bins (int) 
        number of bins for distance thresholding
    Output:
    - G: n x n 
        group connectivity matrix using distance thresholding
    - Gc: n x n 
        group connectivity matrix using classic consensus thresholding
    '''
    
    n, _, nsub = np.shape(A)    # number nodes (n) and subjects (nsub)
    C = np.sum(A > 0, axis=2)   # consistency
    W = np.sum(A, axis=2) / C   # average weight
    W[np.isnan(W)] = 0          # remove nans
    Grp = np.zeros((n, n, 2))   # one dim each for storing inter/intra hemispheric connections
    Gc = Grp.copy()
    
    # create bins based on edge distances
    distbins = np.linspace(np.min(np.nonzero(dist)), np.max(np.nonzero(dist)), nbins + 1)
    distbins[-1] = distbins[-1] + 1
    
    for j in range(2): # calculate G for each intra and inter hemisphere connections seperately
        
        # make inter- or intra-hemispheric edge mask
        if j == 0:                                   # inter 
            d = (hemiid == 0) * (hemiid.T == 1)
            d = np.logical_or(d, d.T)
        else:                                        # intra
            d = (hemiid == 0) * (hemiid.T == 0) | (hemiid == 1) * (hemiid.T == 1)
            d = np.logical_or(d, d.T)
        
        #get indices of all valid connections in matrix, weighted by distance
        D = np.empty_like(A)
        upper_mask = dist * np.triu(d)
        for i in range(nsub):
            binary_A = A > 0
            D[:, :, i] = binary_A[:, :, i] * upper_mask
        D = D[np.nonzero(D)].flatten()
        
        #target number of edges for thresholding
        tgt = len(D) / nsub
        G = np.zeros((n*n))
        
        # For each distance distribution bin
        for ibin in range(nbins):
            
            # find out how much of the connections to keep in current bin
            D_lower = D >= distbins[ibin]
            D_upper = D < distbins[ibin + 1]
            frac = round((tgt * np.sum(np.logical_and(D_lower, D_upper))) / len(D))
            print(frac)
            
            # mask out connections not in current bin to consensus matrix
            c = np.triu(np.logical_and(dist >= distbins[ibin], dist < distbins[ibin + 1])) * C * d
            
            # sort connections in descending order by how frequently they appear in each subject
            sorted_idx = np.argsort(c, axis=None, kind='quicksort', order=None)[::-1]
            print(np.count_nonzero(c))
            
            # keep connections up until frac
            G[sorted_idx[:frac]] = 1
          
        # Distance-based group matrix, for both intra and inter hemisphere connections
        Grp[:, :, j] = np.reshape(G, [n, n])
        
        # Consensus thresholding  
        w = W * np.triu(d)
        idx = np.argsort(w, axis=None, kind='quicksort', order=None)[::-1]
        w = np.zeros((n*n))
        w[idx[:np.count_nonzero(G)]] = 1
        Gc[:, :, j] = np.reshape(w, [n, n])
    
    # add the hemispheres and make the graph symmetric
    G = np.sum(Grp, axis=2)
    G = np.maximum(G, G.transpose())
    
    Gc = np.sum(Gc, axis=2)
    Gc = np.maximum(Gc, Gc.transpose())
        
    return G, Gc
