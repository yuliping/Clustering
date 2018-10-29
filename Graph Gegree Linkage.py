from sklearn import preprocessing
import numpy as np
import scipy.io
import math
import ClusteringMeasure
global MYINF
MYINF = 1e10

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

def BuildAdjacency(CMat, K):
    '''
    This function takes a NxN coefficient matrix and returns a NxN adjacency
    matrix by choosing only the K strongest connections in the similarity
    graph
    :param CMat:NxN coefficient matrix
    :param K:number of strongest edges to keep; if K=0 use all the coefficients
    :return:NxN symmetric adjacency matrix
    '''
    N = CMat.shape[0]
    CAbs = np.abs(CMat)
    for i in range(0,N):
        c = CAbs[:,i]
        PInd = np.argsort(-c) #按照降序排列
        PInd = PInd + 1
        if (np.abs(c[PInd[0] - 1]) != 0):
            CAbs[:,i] = CAbs[:,i] / (np.abs(c[PInd[0] - 1]))
        else:
            CAbs[:, i] = CAbs[:, i] / math.exp(-12)
    CSym = CAbs + CAbs.T
    CSym = CSym/2
    if(K != 0 ):
        Ind = np.argsort(-CSym,axis=0) #表示按列降序排序
        CK = np.zeros((N,N))
        for i in range(0,N):
            for j in range(0,K):
                if (0 == CSym[Ind[0,i],i]):
                    CK[Ind[j, i], i] = CSym[Ind[j, i], i]/math.exp(-12)
                else:
                    CK[Ind[j, i], i] = CSym[Ind[j, i], i] /CSym[Ind[0, i], i]
        CKSym = CK + CK.T
        CKSym = CKSym/2
    else:
        CKSym = CSym
    return CKSym

def makeInitialClusters(y):
    num = len(np.unique(y))
    initialClusters = []
    for i in range(0,num):
        initialClusters.append(y[y == i])
    return initialClusters

def gdlComputeAffity(graphW, cluster_i, cluster_j):
    '''
    计算两个簇之间的关联度

    :param graphW:
    :param cluster_i:
    :param cluster_j:
    :return:
    '''
    num_i = len(cluster_i); num_j = len(cluster_j)
    sum1 = 0
    for j in range(0,num_j):
        inDegree = 0;  outDegree = 0
        index_j = cluster_j[j]
        for i in range(0,num_i):
            index_i = cluster_i[i]
            inDegree = inDegree + graphW[index_i, index_j]
            outDegree = outDegree + graphW[index_j,index_i]
        sum1 = sum1 + inDegree * outDegree
    sum2 = 0
    for i in range(0,num_i):
        inDegree = 0; outDegree = 0
        index_i = cluster_i[i]
        for j in range(0,num_j):
            index_j = cluster_j[j]
            inDegree = inDegree + graphW[index_j,index_i]
            outDegree = outDegree + graphW[index_i,index_j]
        sum2 = sum2 + inDegree*outDegree
    return sum1/(num_i*num_i) + sum2/(num_j*num_j), sum1/(num_i*num_i), sum2/(num_j*num_j)

def gdlInitAffinityTable_c(graphW, initClusters):
    '''

    :param graphW:
    :param initClusters:
    :return:
    '''
    #initialize the affinityTab(asymmetric) and the AsymAffTab(symmetric)
    numClusters = len(initClusters)
    affinityTab, AsymAffTab = np.ones(numClusters*numClusters), np.ones(numClusters*numClusters)
    affinityTab[:], AsymAffTab[:] = -MYINF, -MYINF
    affinityTab = affinityTab.reshape(numClusters,numClusters)
    AsymAffTab = AsymAffTab.reshape(numClusters,numClusters)
    for j in range(0,numClusters):
        cluster_j = initClusters[j]
        for i in range(0,j):
            affinityTab[i,j], AsymAffTab[i,j], AsymAffTab[j,i]= gdlComputeAffity(graphW,initClusters[i], cluster_j)
    return affinityTab, AsymAffTab

def gacPartialMin_triu_c(affinityTab, curGroupNum):
    '''

    :param affinityTab:
    :param curGroupNum:
    :return:
    '''
    numClusters = affinityTab.shape[1]
    if ((2 != affinityTab.ndim) or (numClusters != affinityTab.shape[0])):
        print("affinityTab is not valid!")
        return
    minIndex1, minIndex2 = 0, 0
    minElem = MYINF
    for j in range(0,curGroupNum):
        for i in range(0,j):
            if(affinityTab[i,j] < minElem):
                minElem = affinityTab[i,j]
                minIndex1 = i
                minIndex2 = j
    if(minIndex1 > minIndex2):
        tmp = minIndex1
        minIndex1 = minIndex2
        minIndex2 = tmp
    return minIndex1, minIndex2

def gdlComputeDirectedAffity(graphW, cluster_i, cluster_j):
    '''

    :param graphW:
    :param cluster_i:
    :param cluster_j:
    :return:
    '''
    num_i = len(cluster_i); num_j = len(cluster_j)
    sum1 = 0
    for j in range(0,num_j):
        index_j = cluster_j[j]
        inDegree = 0; outDegree = 0
        for i in range(0, num_i):
            index_i = cluster_i[i]
            inDegree = inDegree + graphW[index_j, index_i]
            outDegree = outDegree + graphW[index_i, index_j]
        sum1 = sum1 + inDegree * outDegree
    return sum1/(num_i*num_i)

def gdlDirectedAffinity_batch_c (graphW, initClusters, minIndex1):
    '''

    :param graphW:
    :param initClusters:
    :param minIndex1:
    :return:
    '''
    #i = minIndex1
    numClusters = len(initClusters) #这个不太确定对不对
    pAsymAffTab = np.zeros((1, numClusters))
    if((graphW.ndim != 2) or (graphW.shape[0] != graphW.shape[1])):
        print("graphW is not a square Matrix")
        return
    for j in range(0,numClusters):
        if(j == minIndex1):
            pAsymAffTab[0,j] = -1e10
        else:
            pAsymAffTab[0,j] = gdlComputeDirectedAffity(graphW,initClusters[minIndex1],initClusters[j])
    return pAsymAffTab

def gdlMerging_c(graphW, initClusters, groupNumber):
    '''

    :param graphW:asymmetric weighted adjacency matrix
    :param initClusters:a cell array of clustered vertices
    :param groupNumber:the final number of clusters
    :return:clusterLabels: 1 x m list whose i-th entry is the group assignment of
            the i-th data vector w_i. Groups are indexed sequentially, starting from 1.
    '''
    numSample = graphW.shape[0]
    myInf = 1e10
    #initialization
    VERBOSE = False
    numClusters = len(initClusters)
    assert numClusters > groupNumber, 'GAC: too few initial clusters. Do not need merging!'
    #compute initial (negative) affinity table (upper trianglar matrix), very slow
    if (VERBOSE):
        np.disp('   Computing initial table.')
    affinityTab, AsymAffTab = gdlInitAffinityTable_c(np.double(graphW), initClusters)
    affinityTab = np.tril(myInf*np.ones((numClusters,numClusters)),k=0) - affinityTab
    if (VERBOSE):
        np.disp('   Starting merging process')
    curGroupNum = numClusters
    while (True):
        if (0 == np.mod(curGroupNum, 50) and VERBOSE):
            np.disp('   Group count: ',str(curGroupNum))
        #find two clusters with the best affinity
        minIndex1,minInedx2 = gacPartialMin_triu_c(affinityTab, curGroupNum)
        cluster1 = initClusters[minIndex1]
        cluster2 = initClusters[minInedx2]
        #merge the two clusters
        cluster1 = np.reshape(cluster1,(len(cluster1), 1))
        cluster2 = np.reshape(cluster2,(len(cluster2), 1))
        new_cluster = np.vstack((cluster1,cluster2))
        #move the second cluster to be merged to the end of the cluster array
        #note that we only need to copy the end cluster's information to
        #the second cluster's position
        if (minInedx2 != curGroupNum):
            initClusters[minInedx2] = initClusters[-1]
            #affinityTab is an upper triangular matrix
            affinityTab[0:minInedx2,minInedx2] = affinityTab[0:minInedx2,curGroupNum-1]
            affinityTab[minInedx2, minInedx2+1:curGroupNum-1] = affinityTab[minInedx2+1:curGroupNum-1,curGroupNum-1]
        AsymAffTab[0:curGroupNum,minIndex1] = AsymAffTab[0:curGroupNum, minIndex1] + AsymAffTab[0:curGroupNum, minInedx2]
        AsymAffTab[0:curGroupNum, minInedx2] = AsymAffTab[0:curGroupNum,curGroupNum-1]
        AsymAffTab[minInedx2,0:curGroupNum] = AsymAffTab[curGroupNum-1,0:curGroupNum]
        #update the first cluster and remove the second cluster
        initClusters[minIndex1] = new_cluster
        initClusters.pop(-1)
        affinityTab[:,curGroupNum-1] = myInf
        affinityTab[curGroupNum-1,:] = myInf
        curGroupNum = curGroupNum - 1
        if curGroupNum <= groupNumber:
            break
        AsymAffTab[minIndex1,0:curGroupNum] = gdlDirectedAffinity_batch_c (graphW, initClusters, minIndex1)
        affinityTab[0:minIndex1,minIndex1] = -AsymAffTab[minIndex1,0:minIndex1].T - AsymAffTab[0:minIndex1, minIndex1]
        affinityTab[minIndex1, minIndex1+1:curGroupNum] = -AsymAffTab[minIndex1,minIndex1+1:curGroupNum] - AsymAffTab[minIndex1+1:curGroupNum,minIndex1].T
    clusterLabels = np.ones((numSample, 1))
    for i in range(0,len(initClusters)):
        clusterLabels[initClusters[i]] = i
    if (VERBOSE):
        np.disp('   Final group count:', str(curGroupNum))
    return clusterLabels