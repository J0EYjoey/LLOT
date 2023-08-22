import numpy as np
import ot

from scipy.special import logsumexp

def LM(X, a, b):
    '''
    X: spatial expression,
    a: linear coefficients
    b: intercepts
    '''
    N = np.shape(X)[0]
    D = np.shape(X)[1]
    Y = np.zeros((N, D))
    for i in range(D):
        Y[:, i] = a[i] * X[:, i] + b[i]

    return Y

def train(Z,X,W):
    '''
    Z: spatial expression
    X: scRNA expression
    W: coupling matrix
    '''
    #start = time.time()
    m, n = Z.shape, X.shape
    x = Z.repeat(n).reshape(-1, 1)
    y = np.tile(X, m)
    w = W.ravel()
    w = w / np.max(w)
    idx = np.where(w>0)[0]
    x = x[idx].flatten()
    y = y[idx]
    w = w[idx]
    w = w/np.sum(w)
    xb = np.dot(w,x)
    yb =  np.dot(w,y)
    vx = np.sum(w*(x**2))-xb**2
    cxy = np.sum(x*w*y)-xb*yb
    alpha = cxy/vx
    beta = yb-alpha*xb
    #end = time.time()
    #print(end-start)
    return alpha,beta

def SLR(x,y):
    xb = np.mean(x)
    yb = np.mean(y)
    vx = np.var(x)
    cxy = np.cov(x,y)[0,1]
    alpha = cxy/vx
    beta = yb-alpha*xb
    return alpha,beta


def stochastic_train(Z, X, W, size=1000):
    '''
    Z: spatial expression
    X: scRNA expression
    W: coupling matrix
    size: stochastic size
    '''
    d = Z.shape[1]
    A = np.zeros(d)
    B = np.zeros(d)

    m, n = pi.shape
    psample = pi.flatten()
    sample_idx = np.random.choice(n * m, size=size, replace=False, p=psample)
    row_idx = sample_idx // n
    col_idx = sample_idx % n
    for i in range(d):
        x = Z[row_idx, i]
        y = X[col_idx, i]
        A[i], B[i] = SLR(x, y)

    return A, B



def LLOT(numitr=20, lambda1=1, lambda2=1, ST=None, RNAseq=None, L=None,  Suggest=False):
    '''
    numitr: number of iteration
    lambda1: entropic parameter
    lambda2: Laplacian parameter
    ST: Spatial transcriptomics data
    RNAseq: scRNA-seq dataset
    L : Laplacian matrix
    Suggest: Use suggested parameters if True, default = False
    
    Return the coupling matrix pi
    '''
    X = RNAseq.copy()
    Z = ST.copy()
    m, n = ST.shape[0], RNAseq.shape[0]
    u = np.zeros(m)
    v = np.zeros(n)
    d = ST.shape[1]
    p = np.ones(m) / float(m)
    q = np.ones(n) / float(n)
    
   
    
    A = np.std(X, axis=0) / np.std(Z, axis=0)
    B = np.mean(X, axis=0) - np.mean(Z, axis=0)
    
    Y = LM(Z, A, B)
    M0 = np.mean(ot.dist(Y, X))
    if(Suggest==True):
        lambd1 = M0 / 50
        lambd2 = M0 / 10
    else:
        lambd1 = lambda1
        lambd2 = lambda2
    for i in range(10):
            # if i % 10 == 0:
            #   print(str(i) + '-th iteration')
        Y = LM(Z, A, B)
        M = ot.dist(Y, X)
        Mr = -M / lambd1

            # update dual
        v = np.log(q) - logsumexp(Mr + u[:, None], 0)
        u = np.log(p) - logsumexp(Mr + v[None, :], 1)

            # current coupling pi
        pi = np.exp(Mr + u[:, None] + v[None, :])

            # update linear map
        for j in range(d):
            U = train(Z[:, j], X[:, j], pi)
            A[j] = U[0]
            B[j] = U[1]
    
    for i in range(numitr):
        A_old = A.copy()
        B_old = B.copy()
        Y = LM(Z, A, B)
        M = ot.dist(Y, X)

        G = 2 * lambd2 * np.dot(L, pi) + M
            # temp = ot.sinkhorn(a,b,M = G, reg=lambd_1, method='sinkhorn_stabilized',numItermax = inner_itr)
            # temp = ot.sinkhorn(p,q,G,reg = lambd1,method='sinkhorn_log',numItermax = 500000)
        
        temp = ot.sinkhorn(p, q, G, reg=lambd1, method='sinkhorn', numItermax=5000, warn=False)
#        else:
#            temp = ot.sinkhorn(p, q, G, reg=lambd1, method='sinkhorn_log', numItermax=sk_itr, warn=False)
        step = 0.5 / (i + 3)
        pi = step * temp + (1 - step) * pi

        for j in range(d):
            A[j], B[j] = train(Z[:, j], X[:, j], pi)

            A = step * A + (1 - step) * A_old
            B = step * B + (1 - step) * B_old
    return pi

    

def LLOT_stochastic(numitr=40, lambda1=1, lambda2=1, ST=None, RNAseq=None,  L=None, Suggest =False, B=1000,seed=666):
    '''
    numitr: number of iteration
    lambda1: entropic parameter
    lambda2: Laplacian parameter
    ST: Spatial transcriptomics data
    RNAseq: scRNA-seq dataset
    L : Laplacian matrix
    Suggest: Use suggested parameters if True, default = False
    B: batch size
    seed : random seed , default =666
    Return the coupling matrix pi
    '''
    np.random.seed(seed)
    m, n = ST.shape[0], RNAseq.shape[0]
    u = np.zeros(m)
    v = np.zeros(n)
    d = ST.shape[1]
    p = np.ones(m) / float(m)
    q = np.ones(n) / float(n)
    gamma = 0.5
    X = RNAseq
    Z = ST
    A = np.std(X, axis=0) / np.std(Z, axis=0)
    B = np.mean(X, axis=0) - np.mean(Z, axis=0)
    
    Y = LM(Z, A, B)
    M0 = np.mean(ot.dist(Y, X))
    
    if(Suggest==True):
        lambd1 = M0 / 50
        lambd2 = M0 / 10
    else:
        lambd1 = lambda1
        lambd2 = lambda2
        
    pi = ot.sinkhorn(p, q, M, reg=lambd1, method='sinkhorn', numItermax=10000) # warm up initial of pi

    for i in range(numitr):
        # print('Iteration ' + str(i+1)+'is ready')
        A_old = A.copy()
        B_old = B.copy()
        M = ot.dist(Y, X)
        G = 2 * lambd1 * (np.dot(L, pi)) + M
        temp = ot.sinkhorn(p, q, G, reg=lambd1, method='sinkhorn', numItermax=10000)
        r = gamma / (i + 1)
        pi = (1 - r) * pi + r * temp
            # pi = ot.sinkhorn(p, q, M,reg=lambd1,numItermax=10000)
            # pi = ot.stochastic.sag_entropic_transport(p, q, M,reg=lambd1,numItermax=10000,lr=0.1)
            # print('regression')
        A, B = stochastic_train(Z, X, pi, size=B)
            # print('regression finished')
        r = gamma / (i + 1)
           
        A = r * A + (1 - r) * A_old
        B = r * B + (1 - r) * B_old
        Y = LM(Z, A, B)
    # otmap_lst.append(pi)
    return pi








