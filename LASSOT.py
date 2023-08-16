import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.decomposition import PCA
import ot
import sys
#from sklearn.linear_model import LinearRegression
from scipy.special import logsumexp
from scipy.sparse import csgraph
from sklearn.neighbors import KNeighborsTransformer


np.random.seed(666)

slide_seq = pd.read_csv("Slide_seq_normalize.csv", sep=',', index_col=0).T
loc = pd.read_csv("slide_seq_loc.csv", sep=',')
loc = loc.values
vargene= slide_seq.columns

scRNA = pd.read_csv('All_Cluster_Cells.csv', sep=',', index_col=0)
scRNA=scRNA.T



transformer2 = KNeighborsTransformer(n_neighbors=10,mode='connectivity')
loc_con_knn = transformer2.fit_transform(loc[:,1:3])

aff = (loc_con_knn+loc_con_knn.T)/2 # NN affinity matrix
L = csgraph.laplacian(aff)
L = L.toarray()


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


col = list(set(scRNA.columns)&set(vargene))


gamma = 0.5





X = scRNA.values
Z = slide_seq.values

d = 150
   # d=128
pca = PCA(n_components=d)

pca.fit(Z)
X = pca.transform(X)
Z = pca.transform(Z)
m, n = Z.shape[0], X.shape[0]

    # lambd1 = lambd_list[s]
pi = np.zeros(shape=(m, n))
p = np.ones(m) / float(m)
q = np.ones(n) / float(n)
    # np.random.seed(666)
    # A = np.ones(shape=d)
    # B = np.zeros(shape=d)
    # Y = Z
A = np.std(X, axis=0) / np.std(Z, axis=0)
B = np.mean(X, axis=0) - np.mean(Z, axis=0)
numitr=40
Y = LM(Z, A, B)
M = ot.dist(Y, X)
lambd1 = np.mean(M) / 100
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
    A, B = stochastic_train(Z, X, pi, size=10000)
        # print('regression finished')
    r = gamma / (i + 1)
       
    A = r * A + (1 - r) * A_old
    B = r * B + (1 - r) * B_old
    Y = LM(Z, A, B)
    # otmap_lst.append(pi)
piname = 'pi.npy'
np.save(file=piname, arr=pi)
np.save(file = 'A.npy',arr=A)
np.save(file = 'B.npy',arr=B)









