import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from scipy import sparse
from sklearn.cluster import KMeans

global eps
eps = np.spacing(1)

def L2_distance_1(a,b):
    '''

    :param a:
    :param b: two matrices.each column is a data
    :return: d: distance matrix of a and b
    '''
    if(1 == a.shape[0]):
        a = np.row_stack((a,np.zeros((1,a.shape[1]))))
        b = np.row_stack((b,np.zeros((1,b.shape[1]))))
    aa  = np.sum(np.multiply(a,a),axis = 0)
    bb = np.sum(np.multiply(b, b), axis=0)
    ab = np.dot(np.transpose(a), b)
    d = np.tile(aa.reshape(aa.shape[0], 1), (1,len(bb))) + np.tile(bb, (len(aa),1)) - 2 * ab
    d= np.real(d)
    d[d < 0] = 0
    return d

def constructW_PKN(*args):
    '''

    :param X: each coloumn is a data point
    :param k: number of neighbors
    :param issymmetric: set W = (W + W')/2 if issymeric = 1
    :return: W:similarity matrix
    '''
    if(len(args) < 3):
        issymmetric = 1
    elif(len(args) < 2):
        k = 5
    elif(len(args) < 1):
        print("This function has no input, please enter the correct parameters")
        return
    else:
        X = args[0]; k = args[1]; issymmetric = args[2]
    dim, n = np.shape(X)
    D = L2_distance_1(X, X)
    idx = np.argsort(D,axis = 1)
    W = np.zeros((n,n))
    for i in range(0,n):
        id = idx[i,1:k+2]
        di = D[i, id]
        W[i, id] = (di[k] - di)/(k*di[k] - sum(di[0:k]) + eps)
        #为什么sum(di[0:k-1])算出来的数据差别比较大？？
    if(1 == issymmetric):
        W = (W + np.transpose(W))/2
    return W

def eig1(*args):
    '''

    :param A:
    :param c:
    :param isMax:
    :param isSym:
    :return:
    '''
    if(len(args) < 1):
        print("This function has no input, please enter the correct parameters")
        return
    else:
        A = args[0]
    if(len(args) < 2):
        c = A.shape[0]
        isMax = 1; isSym = 1
    else:
        c = args[1]
        if(A.shape[0] < c):
            c = A.shape[0]
    if(len(args) < 3):
        isMax = 1; isSym =1
    else:
        isMax = args[2]
    if(len(args) < 4):
        isSym = 1
    else:
        isSym = args[3]
    if(isSym == 1):
        A = np.maximum(A, np.transpose(A))
    d, v = np.linalg.eig(A)
    if(0 == isMax):
        idx = np.argsort(d)   #应该要索引值，不是要距离的排序，注意与matlab的区别
    else:
        idx = np.argsort(-d)
    idx1 = idx[0:c]
    eigval = d[idx1]
    eigvec = v[:,idx1]
    eigval_full = d[idx]
    return eigvec,eigval_full

def EProjSimplexdiag(d, u):
    '''
    This function is to solve the problem:
    min 1/2*x'*Ux - x'd
    s.t. x>=0, 1'x= 1

    :param d:
    :param u:
    :return:
    '''
    lamda = (u-d).min()
    f = 1
    count = 1
    while(abs(f) > 10**(-8)):
        v1 = 1/u * lamda + d/u
        posidx = v1 > 0
        g = np.sum(1/u[posidx], axis = 0)
        f = np.sum(v1[posidx]) - 1
        lamda = lamda - f/g
        if(count > 1000):
            break
        count +=1
    v1 = 1/u * lamda + d/u
    x = np.where(v1 > 0, v1, 0)
    return x

def  EProjSimplex_new(args):
    '''
    This funvtion is to solve the equation:
    min 1/2 || x-v||^2 s.t. x>=0,1'x = k
    :param v:
    :param k:
    :return:
    '''
    if(len(args) < 1):
        print("This function has no input, please enter the correct parameters")
        return
    else:
        v = args[0]
    if(len(args) < 2):
        k = 1
    else:
        k = args[1]
    ft = 1
    n = np.size(v, axis = 0)
    v0 = v - np.mean(v) + k/n
    vmin = np.min(v0)
    if(0 >vmin):
        f = 1
        lambda_m = 0
        while(abs(f) > 10**(-10)):
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx,axis = 0)
            g = -npos
            f = np.sum(v1[posidx],axis=0) - k
            lambda_m = lambda_m - f/g
            ft = ft + 1
            if(ft > 100):
                x = np.where(v1 > 0, v1, 0)
        x = np.where(v1 > 0, v1, 0)
    else:
        x = v0
    return x

def CLR(A0, c, isrobust, islocal):
    '''
    :param A0: the given affinity matrix
    :param c: cluster number
    :param isrobust: solving the second problem if isrobust = 1
    :param islocal: only update the similarities of neighbors if islocal = 1
    :return:y,S,evs,cs
    '''
    NITER = 30; zr = 10 * math.exp(-11); lamda = 0.1; r = 0
    A0 = A0 - np.diag(np.diag(A0))
    num = A0.shape[0]
    A10 = (A0 + np.transpose(A0))/2
    D10 = np.diag(A10.sum(axis = 0))
    L0 = D10 - A10
    #automatically determine the cluster number
    [F0, evs] = eig1(L0, num, 0)
    a = np.abs(evs); a[a < zr] = eps; ad = np.diff(a)
    ad1 = ad/a[1:len(a)]
    ad1[ad1 > 0.85] = 1
    ad1 = ad1 + eps * np.transpose(np.arange(1,num))
    ad1[0] = 0
    ad1 = ad1[0:math.floor(0.9*len(ad1))]
    cs = np.argsort(-ad1)
    formated_str = "Suggested cluster number is: %d, %d, %d, %d, %d"%(cs[0]+1, cs[1]+1, cs[2]+1,cs[3]+1,cs[4]+1)
    print(formated_str)
    F = F0[:,0:c]
    if(zr > np.sum(evs[0:c+1],axis = 0)):
        formated_str = "The original graph has more than %d connected component" %(c)
        print(formated_str)
        return
    if (zr > np.sum(evs[0:c + 1], axis=0)):
        clusternum, y = connected_components(coo_matrix(A10), connection='strong')
        y = np.transpose(y)
        S = A0
        return
    u = []
    for i in range(0,num):
        a0 = A0[i,:]
        if(1 == islocal):
            idxa0 = np.array(np.where(a0>0))
        else:
            idxa0 = np.arange(0,num)
        u.append(np.ones((1,idxa0.shape[1])))
    for iter in range(0,NITER):
        dist = L2_distance_1(np.transpose(F), np.transpose(F))
        S = np.zeros((num,num))
        for i in range(0,num):
            a0 = A0[i,:]
            if(1 == islocal):
                idxa0 = np.array(np.where(a0>0))
            else:
                idxa0 = np.arange(1,num)
            ai = a0[idxa0]
            di = dist[i,idxa0]
            if(1 == isrobust):
                for ii in range(0,1):
                    ad = u[i] * ai -lamda*di/2
                    si = EProjSimplexdiag(ad, u[i] + r*np.ones((1,len(idxa0))))
                    u[i] = 1/(2*np.sqrt((si - ai)**2 + eps))
                S[i,idxa0] = si
            else:
                ad = ai - 0.5*lamda*di
                S[i,idxa0] =  EProjSimplex_new(ad)
        A = S
        A = (A + np.transpose(A))/2
        D = np.diag(np.sum(A,axis=0))
        L = D - A
        F_old = F
        F, ev = eig1(L, c, 0)
        evs = np.vstack((evs, ev))
        fn1 = np.sum(ev[0:c],axis=0)
        fn2 = np.sum(ev[0:c+1], axis=0)
        if(fn1 > zr):
            lamda = 2 * lamda
        elif(fn2 < zr):
            lamda = lamda/2; F = F_old
        else:
            break
    clusternum, y = connected_components(coo_matrix(A),connection='strong')
    y = np.transpose(y)
    if(clusternum != c):
        formated_str = "Can not find the correct cluster number :%d" %(c)
        print(formated_str)
    return y,S,evs,cs

def tuneKmeans(M, Ini):
    '''

    :param M:
    :param Ini:
    :return:
    '''
    minob = 10**5
    nIni = Ini.shape[1]
    kmobj = np.zeros((nIni,nIni))
    for ii in range(0,nIni):
        result = KMeans(n_clusters=4).fit((M))
        Ind = result.labels_; obj = result.inertia_
        if(obj < minob):
            minob = obj
            finalInd = Ind
    return finalInd


'''
def twomoon_gen(num1, num2, sigma_noise, horizonal, vertical, *args):
    if(1 == len(args)):
        num2 = num1
    if(2 >= len(args)):
        sigma_noise = 0.12
    if (3 >= len(args)):
        level = 0.35
        upright = 0.15
    else:
        level = 0.32 + horizonal
        upright = 0.15 + vertical

    i = 0; j = 0
    cosT = np.zeros((num1))
    sinT = np.zeros((num1))
    while(j < num1):
        i = i + ((math.pi)/(num1-1))
        cosT[j] = math.cos(i)
        sinT[j] = math.sin(i)
        j = j + 1
'''

if __name__ == '__main__':
    '''
    测试数据集一
    '''
    datasettoy = np.loadtxt('toy.txt')
    n = 100; c = 4; n1 = n/c; A = copy.deepcopy(datasettoy)
    A = A - np.diag(np.diag(A))
    A0 = A
    A = (A + A.T)/2
    y0 = np.concatenate((np.ones(25), 2 * np.ones(25)))
    y0 = np.concatenate((y0, 3 * np.ones(25)))
    y0 = np.concatenate((y0, 4 * np.ones(25)))
    fig = plt.figure()
    plt.imshow(A,cmap = 'jet')
    plt.colorbar()

    isrobust = 0
    y, S, evs, cs = CLR(A0, c, isrobust, 1)
    y = y + 1

    fig = plt.figure()
    plt.imshow(S, cmap='jet')
    plt.colorbar()

    isrobust = 1
    y, S, evs, cs = CLR(A0, c, isrobust, 1)
    fig = plt.figure()
    plt.imshow(S, cmap='jet')
    plt.colorbar()

    #RCut & NCut
    D = np.diag(np.sum(A,axis = 0))
    nRepeat = 100
    Ini = []
    Tni = np.zeros((n, nRepeat))
    for jj in range(0,nRepeat):
        if(0 == jj):
            Ini = np.random.choice(range(1,c+1),100)
        else:
            Ini = np.row_stack((Ini, np.random.choice(range(1, c + 1), 100)))

    #RCut
    print("Ratio Cut\n")
    Fg, tmpD = eig1((D-A), c, 0, 1)
    sqrtSum = np.sqrt(np.sum(Fg**2,axis = 1))
    sqrtSum = sqrtSum.reshape(sqrtSum.shape[0],1)
    Fg = Fg/np.kron(np.ones((1,c)),sqrtSum)
    y = tuneKmeans(Fg, Ini)

    #NCut
    print("Normalized Cut\n")
    Dd = np.diag(D)
    Dn = sparse.spdiags(np.sqrt(1/Dd), 0, n, n)
    An = Dn * A * Dn
    An = (An + An.T)/2
    Fng,D = eig1((An),c,1,1)
    sqrtSum = np.sqrt(np.sum(Fg ** 2, axis=1))
    sqrtSum = sqrtSum.reshape(sqrtSum.shape[0], 1)
    Fng = Fng / np.kron(np.ones((1, c)), sqrtSum)
    y = tuneKmeans(Fng,Ini)
    '''
    测试数据集二
    '''
    num0 = 100
    datasets = np.loadtxt('twomoon.txt')
    y0 = np.concatenate((np.ones(100), 2*np.ones(100)))
    y0 = y0.reshape(y0.shape[0], 1)
    c= 2 #the number of cluster
    D = L2_distance_1(np.transpose(datasets), np.transpose(datasets))
    A0 = constructW_PKN(np.transpose(datasets), 10, 0)
    A = A0
    A = (A + np.transpose(A))/2
    la = y0

    #The original data
    fig = plt.figure()
    plt.plot(datasets[:, 0], datasets[:, 1], 'r.', markersize = 8)
    plt.plot(datasets[0:la[la==1].shape[0],0], datasets[0:la[la==1].shape[0], 1], 'r.', markersize = 8)
    plt.plot(datasets[0:la[la == 2].shape[0], 0], datasets[0:la[la == 2].shape[0], 1], 'b.', markersize = 8)
    plt.axis('equal')
    fig = plt.figure()
    plt.plot(datasets[:, 0], datasets[:, 1], 'r.')
    plt.plot(datasets[0:la[la == 1].shape[0], 0], datasets[0:la[la == 1].shape[0], 1], 'r.', markersize = 8)
    plt.plot(datasets[0:la[la == 2].shape[0], 0], datasets[0:la[la == 2].shape[0], 1], 'b.', markersize = 8)
    nn = 2*num0
    for i in range(0,nn):
        for j in range(0,i):
            weight = A[i, j]
            if (weight > 0):
                plt.plot([datasets[i, 0], datasets[j, 0]], [datasets[i, 1], datasets[j, 1]], color = "green", linewidth = 10*weight, linestyle = '-', markersize = 8)
    plt.axis('equal')
    y, S, evs, cs = CLR(A0, c, 0, 1)
    A = (S + S.T)/2
    la = y
    #Adaptive neighbors with L2, line width denotes similarity
    fig = plt.figure()
    plt.plot(datasets[:,0], datasets[:,1], 'k.', markersize = 8)
    plt.plot(datasets[0:la[la == 1].shape[0], 0], datasets[0:la[la == 1].shape[0], 1], 'r.', markersize = 8)
    plt.plot(datasets[0:la[la == 2].shape[0], 0], datasets[0:la[la == 2].shape[0], 1], 'b.', markersize = 8)
    nn = 2 * num0
    for i in range(0, nn):
        for j in range(0, i):
            weight = A[i, j]
            if (weight > 0):
                plt.plot([datasets[i, 0], datasets[j, 0]], [datasets[i, 1], datasets[j, 1]], color="green",
                         linewidth=10 * weight, linestyle='-')
    plt.axis('equal')