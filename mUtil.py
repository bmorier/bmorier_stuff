# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:31:41 2018

@author: bmori
"""

from threading import Thread
from numba import jit,njit, void, float64,boolean,int64
import numba as nb
import numpy as np
import scipy.stats  
import time
import pickle
#@jit(nb.types.Tuple((boolean[:],float64[:]))(float64[:],float64[:]))
#@jit(int64[:](float64[:],float64[:]), nopython=True)
#@jit( nopython=True)

# 
binOp = {'plus':0,'minus':1, 'times':2, 'div' : 3,'or':4,'and':5,'gt':6,'lt':7,'geq':8,'leq':9,'power':10,'min':11,'max':12,'eq':13,'neq':14}

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))
        
def runThreads(wrapped):
    def inner(*args,**kwargs):
        #nonlocal argO
        if 'NThreads' in kwargs:
            NThreads = kwargs['NThreads']
            kwargs.pop('NThreads')
        else:
            NThreads =10
            
        if 'axis' in kwargs:
            axis = kwargs['axis']
            kwargs.pop('axis')
        else:
            axis =0;
        def getM(X):
#            nonlocal axis,i,n
            if X.shape[axis]>1:
                return X[:,:,i*n:(i+1)*n]
            else:
                return X
            
        N = max([arg.shape[axis] for arg in args])
        
        n=int(np.ceil(N/NThreads))
        thread=list()
        for i in range(NThreads):
            args1 = []
            kwargs1 = {}
            for arg in args:
                args1.append(getM(arg))
            
            for key in kwargs:
                kwargs1[key] = getM(kwargs[key])
                
            thread.append(Thread(target =wrapped, args = args1,kwargs =kwargs1 ))
            thread[-1].start()

        for i in range(NThreads):
            thread[i].join()  
        
    return inner
    

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def trimmed(func,delta):
    """
    decorator like to return a function that first trimm data
    
    """
    def inner(*args,**kwargs):
        args1=list()
        for v in args:
            assert(isinstance(v,np.ndarray))
            min1,max1=np.percentile(v,[delta,100-delta])
            v1=v
            v1 = v1[np.logical_and(v1>=min1,v1<=max1)]
            args1.append(v1)
        
        kwargs1=dict()
        for v in kwargs:
            assert(isinstance(kwargs[v],np.ndarray))
            min1,max1=np.percentile(v,[delta,100-delta])
            v1=kwargs[v]
            v1 = v1[np.logical_and(v1>=min1,v1<=max1)]
            kwargs1[v] = v1
        
        return func(*args1,**kwargs1)
    
    return inner

@njit
def std_t(v,delta=10):
    """
    decorator like to return a function that first trimm data
    
    """
    v1=np.sort(v)
    ii=int(v1.size*float(delta)/100)
    v1=v1[ii:-ii]
    return np.std(v1)


@njit
def mean_t(v,delta=10):
    """
    decorator like to return a function that first trimm data
    
    """
    v1=np.sort(v)
    ii=int(v1.size*float(delta)/100)
    v1=v1[ii:-ii]
    return np.mean(v1)


def mvnpdfNP(x,mu,Q):
    return mvnpdf(x,mu,Q)
    
def mvnpdf(x,mu,Q):
 #   Sx= x.shape;
    k=float(x.shape[0]);

    sQ=Q.shape;

    iQ_ = np.reshape(Q,[sQ[0],sQ[1],np.prod(sQ[2:])]);
    lD=np.zeros((1,iQ_.shape[2]));
    if iQ_.shape[0]==1:
        iQ_=(iQ_**0.5);
        lD = np.log(iQ_.flat).T;
        iQ_=1/iQ_;
    else:
#          for i in range(iQ_.shape[2]):
        R = cholesky(iQ_);
        iQ_ = inv(R)
        DD=np.diagonal(R,axis1=0,axis2=1)
        lD = np.sum(np.log(DD),1);##CHECK
        
    iQ_=np.reshape(iQ_,sQ);
    lD = np.reshape(lD,(1,)+sQ[2:]);
    xu=multiprod(iQ_,(x-mu),[0,1],0);
    #xu=multiprod(iQ_,reshape((x-mu),[Sx(1),prod(Sx(2:end))]),[1 2],1);
    if len(xu.shape)>=4:
        xu=np.reshape(xu,[xu.shape[1],xu.shape[2],xu.shape[3]]);

    lp = np.log((2*np.pi))*(-k/2)-lD-multiprod(xu,xu,0,0)/2;
    return lp
    


def mvnpdf1(x1,mu1,Q1):
 #   Sx= x.shape;
    k1=float(x1.shape[0]);

    sQ1=Q1.shape;

    iQ1_ = np.reshape(Q1,[sQ1[0],sQ1[1],np.prod(sQ1[2:])]);
    lD1=np.zeros((1,iQ1_.shape[2]));
    if iQ1_.shape[0]==1:
        iQ1_=(iQ1_**0.5);
        lD1 = np.log(iQ1_.flat).T;
        iQ1_=1/iQ1_;
    else:
#          for i in range(iQ1_.shape[2]):
        R1 = T(cholesky(iQ1_));
        iQ1_ = inv(R1)
        DD1=np.diagonal(R1,axis1=0,axis2=1)
        lD1 = np.sum(np.log(DD1),1);##CHECK
        
    iQ1_=np.reshape(iQ1_,sQ1);
    lD1 = np.reshape(lD1,(1,)+sQ1[2:]);
    xu1=multiprod(iQ1_,(x1-mu1),[0,1],0);
    #xu=multiprod(iQ1_,reshape((x-mu),[Sx(1),prod(Sx(2:end))]),[1 2],1);
    if len(xu1.shape)>=4:
        xu1=np.reshape(xu1,[xu1.shape[1],xu1.shape[2],xu1.shape[3]]);

    lp1 = np.log((2*np.pi))*(-k1/2)-lD1-multiprod(xu1,xu1,0,0)/2;
    return lp1 

def ap1(A,n):
    if n>0:
        return A.reshape(A.shape+(1,)*n,order='F')
    elif n<0:
        return A.reshape((1,)*abs(n)+A.shape,order='F')
    else:
        return A
    
def ap1M(A,i=None):
    if i is None:    
        return A.reshape(A.shape[:-1] + (1,)+A.shape[-1:],order='F')
    else:
        return A.reshape(A.shape[:i] + (1,)+A.shape[i:],order='F')

def repmat(A,sz):
    if not isinstance(A,np.ndarray):
        A=np.array(A)
    if len(sz) > len(A.shape):
        A = ap1(A,len(sz)-len(A.shape))
    return np.tile(A,sz)

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interpNaN(y,mode):
    y1=y.copy(order='A')
    nans, x= nan_helper(y)
    y1[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y1


@jit
def sub2ind(idx,sz):
    assert(len(idx.shape)<=2)
    if len(idx.shape)==1:
        idx=ap1(idx,1)
    Nsz=idx.shape[0]
    assert(Nsz==len(sz))
    
    indOut=np.zeros(idx.shape[-1],dtype=int,order='F');
    
    for i in range(Nsz-1,0,-1):
        indOut = sz[i-1]*(indOut + np.minimum(idx[i,:],sz[i]-1));
    
    indOut =indOut +np.minimum(idx[0,:], sz[0] - 1);
    return indOut;

def ndgrid(nargout,*args):
    
    """
    NDGRID Rectangular grid in N-D space
    %   [X1,X2,X3,...] = NDGRID(x1gv,x2gv,x3gv,...) replicates the grid vectors 
    %   x1gv,x2gv,x3gv,...  to produce the coordinates of a rectangular grid 
    %   (X1,X2,X3,...).  The i-th dimension of the output array Xi are copies
    %   of elements of the grid vector xigv. For example, the grid vector x1gv 
    %   forms the rows of X1, the grid vector x2gv forms the columns of X2 etc. 
    %
    %   [X1,X2,...] = NDGRID(xgv) is equivalent to [X1,X2,...] = NDGRID(xgv,xgv,...).
    %   The dimension of the output is determined by the number of output
    %   arguments. X1 = NDGRID(xgv) degenerates to produce a 1-D grid represented
    %   by a 1-D array.
    %
    %   The coordinate arrays are typically used for the evaluation of functions 
    %   of several variables and for surface and volumetric plots.
    %
    %   NDGRID and MESHGRID are similar, though NDGRID supports 1-D to N-D while 
    %   MESHGRID is restricted to 2-D and 3-D. In 2-D and 3-D the coordinates 
    %   output by each function are the same, the difference is the shape of the 
    %   output arrays. For grid vectors x1gv, x2gv and x3gv of length M, N and P 
    %   respectively, NDGRID(x1gv, x2gv) will output arrays of size M-by-N while 
    %   MESHGRID(x1gv, x2gv) outputs arrays of size N-by-M. Similarly,
    %   NDGRID(x1gv, x2gv, x3gv) will output arrays of size M-by-N-by-P while 
    %   MESHGRID(x1gv, x2gv, x3gv) outputs arrays of size N-by-M-by-P.
    %
    %   Example: Evaluate the function  x2*exp(-x1^2-x2^2-x^3) over the
    %            range  -2 < x1 < 2,  -2 < x2 < 2, -2 < x3 < 2,
    %
    %       [x1,x2,x3] = ndgrid(-2:.2:2, -2:.25:2, -2:.16:2);
    %       z = x2 .* exp(-x1.^2 - x2.^2 - x3.^2);
    %       slice(x2,x1,x3,z,[-1.2 .8 2],2,[-2 -.2])
    %
    %
    %   Class support for inputs x1gv,x2gv,x3gv,...
    %      float: double, single
    %      integer: uint8, int8, uint16, int16, uint32, int32, uint64, int64
    %
    %   See also MESHGRID, SLICE, INTERPN.
    
    """
    if not args or (len(args) > 1 and nargout > len(args)):
       raise('ndgrid:NotEnoughInputs');    

    nout = max(nargout,len(args));
    vout = list()
    if len(args)==1:
        if nargout < 2:
            vout.append(args[0].flatten());    
            return vout
        else:
            j = np.zeros((nout,),dtype=int);
            siz = repmat(args[0].size,[nout,])
            
    else:
        j = np.arange(nout);
        siz = np.array([k.size for k in args]);
    
    #vout = cell(1,max(nargout,1));
    if nout == 2: # Optimized Case for 2 dimensions
        x = ap1(args[j[0]].flatten(),1);
        y = ap1(args[j[1]].flatten(),1).T;
        vout.append(repmat(x,y.shape));
        vout.append(repmat(y,x.shape));
    else:
        for i in range(max(nargout,1)):
            x = args[j[i]];
            s = np.ones((nout,),dtype=int); 
            s[i] = x.size;
            x = np.reshape(x,s);
            s = siz.copy(); 
            s[i] = 1;
            vout.append(repmat(x,s));
    return vout

def GaussHermite(n):
    """
     This function determines the abscisas (x) and weights (w) for the
     Gauss-Hermite quadrature of order n>1, on the interval [-INF, +INF].
        % This function is valid for any degree n>=2, as the companion matrix
        % (of the n'th degree Hermite polynomial) is constructed as a
        % symmetrical matrix, guaranteeing that all the eigenvalues (roots)
        % will be real.
        
        
    % © Geert Van Damme
    % geert@vandamme-iliano.be
    % February 21, 2010    
    """
# Building the companion matrix CM
   #  CM is such that det(xI-CM)=L_n(x), with L_n the Hermite polynomial
    # under consideration. Moreover, CM will be constructed in such a way
    # that it is symmetrical.
    i   = np.arange(1.0,n);
    a   = np.sqrt(i/2);
    CM  = np.diag(a,1) + np.diag(a,-1);
    
    # Determining the abscissas (x) and weights (w)
        # - since det(xI-CM)=L_n(x), the abscissas are the roots of the
        #   characteristic polynomial, i.d. the eigenvalues of CM;
        # - the weights can be derived from the corresponding eigenvectors.
    L,V   = np.linalg.eigh(CM);
    ind = np.argsort(L);
    x=L[ind]
    
    V       = V[:,ind].T;
    w       = np.sqrt(np.pi) * V[:,0]**2;
    return x,w    

    
    


def fortranLA(wrapped):
    def convArg(arg_):
        l=len(arg_.shape)
        ord1 = list(range(2,l))+[0,1]
        return np.transpose(arg_,ord1)
    def inner(*args,**kwargs):
        args1 = []
        kwargs1 = {}
        k=0
        for arg in args:
            args1.append(convArg(arg))
            k=k+1
        
        for key in kwargs:
            kwargs1[key] = convArg(kwargs[key])
            k=k+1
            
        A=wrapped(*args1,**kwargs1)
        def untransp(A):
            l=len(A.shape)
            if l>=3 and l==len(args[0].shape):
                ord1 = [l-2,l-1] + list(range(l-2))
                return np.transpose(A,ord1)
            else:
                return A
        if isinstance(A,tuple):
            B=list()
            for i in range(len(A)):
                B.append(untransp(A[i]))
            return B
        else:
            return untransp(A)
    return inner     
        
@fortranLA
def cholesky(A):
    return np.linalg.cholesky(A)

def chol(A):
    return cholesky(A)

@fortranLA
def inv(A):
    return np.linalg.inv(A)


@fortranLA
def solve(A,B):
    return np.linalg.solve(A,B)

@fortranLA
def pinv(A):
    return np.linalg.pinv(A)

@fortranLA
def svd(A):
    U, S,V= np.linalg.svd(A)
    return U,S.T,V

@fortranLA
def eig(A):
    return np.linalg.eig(A)

@fortranLA
def eigh(A):
    return np.linalg.eigh(A)

@fortranLA
def det(A):
    return np.linalg.det(A)




def cdot(A,B):
    return np.einsum('ijk,ikl->ijl',A,B)

def dot(A,B):
    return np.einsum('ijk,jlk->ilk',A,B)

def matmul(A1,B1,ta=False,tb=False):
    if ta:
        A=T(A1)
    else:
        A=A1

    if tb:
        B=T(B1)
    else:
        B=B1

    return np.einsum('ij...,jl...->il...',A,B)

def cat(axis, args):
    return np.concatenate(args,axis=axis)

def c2cube(m1):
    if len(m1.shape)<3:
        return ap1(m1,1)

def bsxfun(A,B,op):
    
    op = binOp[op]
    
    #binOp = {'plus':0,'minus':1, 'times':2, 'div' : 3,'or':4,'and':5,'gt':6,'lt':7,'geq':8,'leq':9,'power':10,'min':11,'max':12,'eq':13,'neq':14}
    if op==0:
        return A+B
    elif op==1:
        return A-B
    elif op==2:
        return A*B
    elif op==3:
        return A/B
    elif op==4:    
        return np.logical_or(A,B)
    elif op==5:
        return np.logical_and(A,B)
    elif op==6:    
        return A>B
    elif op==7:  
        return A<B
    elif op==8:    
        return A>=B
    elif op==9:    
        return A<=B
    elif op==10:    
        return A**B
    elif op==11:    
        return np.minimum(A,B)
    elif op==12:    
        return np.maximum(A,B)
    elif op==13:    
        return A==B
    elif op==14:    
        return A!=B
    else:
        raise Exception('Invalid Op')



def multiprod(A1,B1,sA,sB,ta=False,tb=False):
 #   assert(len(A.shape)==len(B.shape))
#    I=np.array(A.shape[-sA])!=np.array(A.shape[-sB])
#    assert(np.all(np.logical_or(sA[I]==1,sB[I]==1 )))
    
    if ta:
        A=T(A1)
    else:
        A=A1

    if tb:
        B=T(B1)
    else:
        B=B1
 
    if not isinstance(sA, list):
        sA=[sA,]

    if not isinstance(sB, list):
        sB=[sB,]
            
    strBase = list("abcdefghmnopq");

    inter = len(A.shape)-len(sA)
    assert(inter == len(B.shape)-len(sB))
    strA = strBase[0:inter]
    strB = strBase[0:inter]
    strC = strBase[0:inter]
    
    if len(sA)==1:
        strA.insert(sA[0],'k') 
    else:
        strA.insert(sA[0],'j') 
        strA.insert(sA[1],'k') 


    if len(sB)==1:
        strB.insert(sB[0],'k') 

    else:
        strB.insert(sB[0],'k') 
        strB.insert(sB[1],'l') 

    if len(sA)==2:
        strC.insert(sA[0],'j') 

    if len(sB)==2:
        strC.insert(sB[1],'l') 
        
    strA="".join(strA)
    strB="".join(strB)
    strC="".join(strC)
    return np.einsum(strA+','+strB+'->'+strC ,A,B)


def T(A):
    l=len(A.shape)
    return np.transpose(A,[1,0]+list(range(2,l)))

@fortranLA
def mvnpdfScipy(A):
    return np.log(scipy.stats.multivariate_normal.pdf(A))

        
#@jit(int64[:](float64[:],float64[:]),nopython=True)
@jit(nopython=True)
def _ismemberSorted1(v1,v2):
    """
        assume inputs are sorted
    """
    J=np.empty(v1.size,dtype=np.int64)
    t0=0
    t=0
    for i in range(0,v1.size):
        for t in range(t0,v2.size):
            if v1.flat[i]==v2.flat[t]:
                J[i]=t
                break
            elif v1.flat[i]<v2.flat[t] or t==v2.size-1:
                J[i]=-1
                break
        t0=t
    return J    

def _closestSorted1(v1,v2):
    """
        assume inputs are sorted
    """
    J=np.empty(v1.size,dtype=np.int64)
    t0=0
    t=0
    for i in range(0,v1.size):
        for t in range(t0,v2.size):
            if v1.flat[i]<=v2.flat[t] or t==v2.size-1:
                if t==1 or (abs(v1.flat[i]-v2.flat[t]) < abs(v1.flat[i]-v2.flat[t-1])):
                    J[i]=t
                else:
                    J[i]=t-1
                break
        t0=t
    return J    


#@jit(nb.types.Tuple((boolean[:],int64[:]))(float64[:],float64[:]),nopython=True)
@jit(nopython=True)
def _ismemberSorted(v1,v2):
    J = _ismemberSorted1(v1,v2)
    I=J>=0
    return I,J


#@jit(nb.types.Tuple((boolean[:],int64[:]))(float64[:],float64[:]))
@jit(nopython=True)
def ismember(v1,v2):
    i1 = np.argsort(v1)
    i2 = np.argsort(v2)
    
    v1_ = v1.take(i1)
    v2_ = v2.take(i2)
    z1 = np.argsort(i1) #esse nao precisava se o sort do python fosse decente!

    J_= _ismemberSorted1(v1_,v2_)
    
    J1 = J_[z1 ]    
    J=np.empty(v1.size,dtype=np.int64)
    J[J1>=0]=i2[J1[J1>=0]]    
    I=J>=0
    return I,J

#@jit(int64[:](float64[:],float64[:]))

@jit(nopython=True)
def closest(v1,v2):
    i1 = np.argsort(v1)
    i2 = np.argsort(v2)
    
    v1_ = v1.take(i1)
    v2_ = v2.take(i2)
    z1 = np.argsort(i1) #esse nao precisava se o sort do python fosse decente!

    J_= _closestSorted1(v1_,v2_)
    
    J1 = J_[z1 ]    
    J=np.empty(v1.size,dtype=np.int64)
    J[J1>=0]=i2[J1[J1>=0]]    
    return J


#    v1_,z1 = np.unique(v1,False,True)
#    v2_,i2 = np.unique(v2,True)
    
#    J1 = J_[z1 ]
#    J=np.empty(v1.size,dtype=np.int64)
#    J[J1>=0]=i2[J1[J1>=0]]
#    J[J1<0]=-1
#    I=J>=0
#    return I,J

def testeMatching():
    v1 = np.array([1.,2,2,5,7,8,9,10])
    v2 = np.array([5.,7,7,8,9,10,11,12])
    
    J = _ismemberSorted(v1,v2)
    I=J>=0


def ismemberA(a_vec, b_vec):
    """ MATLAB equivalent ismember function """

    bool_ind = np.isin(a_vec,b_vec)
    common = a_vec[bool_ind]
    common_unique, common_inv  = np.unique(common, return_inverse=True)     # common = common_unique[common_inv]
    b_unique, b_ind = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    return bool_ind, common_ind[common_inv]    
    
def multiprod_(A, B):
    """
    Inspired by MATLAB multiprod function by Paolo de Leva. A and B are
    assumed to be arrays containing M matrices, that is, A and B have
    dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each matrix
    in A with the corresponding matrix in B, using matrix multiplication.
    so multiprod(A, B) has dimensions (M, N, Q).
    """
 
    # First check if we have been given just one matrix
    if len(np.shape(A)) == 2:
        return np.dot(A, B)
 
    # Old (slower) implementation:
    # a = A.reshape(np.hstack([np.shape(A), [1]]))
    # b = B.reshape(np.hstack([[np.shape(B)[0]], [1], np.shape(B)[1:]]))
    # return np.sum(a * b, axis=2)
 
    # Approx 5x faster, only supported by numpy version >= 1.6:
    return np.einsum('ijk,ikl->ijl', A, B)
 
 
def multitransp(A):
    """
    Inspired by MATLAB multitransp function by Paolo de Leva. A is assumed to
    be an array containing M matrices, each of which has dimension N x P.
    That is, A is an M x N x P array. Multitransp then returns an array
    containing the M matrix transposes of the matrices in A, each of which
    will be P x N.
    """
    # First check if we have been given just one matrix
    if A.ndim == 2:
        return A.T
    return np.transpose(A, (0, 2, 1))
 
 
def multisym(A):
    # Inspired by MATLAB multisym function by Nicholas Boumal.
    return 0.5 * (A + multitransp(A))
 
 
def multieye(k, n):
    # Creates a k x n x n array containing k (n x n) identity matrices.
    return np.tile(np.eye(n), (k, 1, 1))
 
 
def multilog(A, pos_def=False):
    # Computes the logm of each matrix in an array containing k positive
    # definite matrices. This is much faster than scipy.linalg.logm even
    # for a single matrix. Could potentially be improved further.
    if pos_def:
        l, v = np.linalg.eigh(A)
        l = np.expand_dims(np.log(l), axis=-1)
        return multiprod(v, l * multitransp(v))
    else:
        raise NotImplementedError
 
 
def multiexp(A, sym=False):
    # Compute the expm of each matrix in an array of k symmetric matrices.
    # Sometimes faster than scipy.linalg.expm even for a single matrix.
    if sym:
        l, v = np.linalg.eigh(A)
        l = np.expand_dims(np.exp(l), axis=-1)
        return multiprod(v, l * multitransp(v))
    else:
        raise NotImplementedError    



class GPUArray(object):
    pass

#
#
#m1=np.ones((5,5), order='F')
#m2=np.ones((5,5),dtype=np.float32, order='F')
#c1=np.ones((5,5,5), order='F')
#c2=np.ones((5,5,5),dtype=np.float32, order='F')
#
#m1_ = array2mat(m1)
#m2_ = array2fmat(m2)
#
#c1_ = array2cube(c1)
#c2_ = array2fcube(c2)
#
#_libarma.testPython(m1_,m2_,c1_,c2_)
#
#
#@arma('mat','fmat','cube','fcube')
#def testPython(m1,m2,c1,c2):
#    _libarma.testPython(m1,m2,c1,c2)
#
#def testPython1(m1,m2,c1,c2):
#    m1_ = array2mat(m1)
#    m2_ = array2fmat(m2)
#    c1_ = array2cube(c1)
#    c2_ = array2fcube(c2)
#    _libarma.testPython(m1_,m2_,c1_,c2_)
#    
#
#
#
##    testPython1(m1,m2,c1,c2)


    
        

    