import matplotlib.pylab as plt
from scipy.special import *
import numpy as np
from math import pi
from numpy import linalg as LA

#Initialize the constants
hbar = 2**(-2)
N = 100
sgm = 4
n0 = 0
A = 0.5*hbar**2/sgm**2
t = 30
#K = 5
#nt = 5

#Preparing the time evolution operator

def matrixN(nt,K):
    matrix=[] #define empty matrix

    for i in xrange(-N,N-1): #total row is 3
        row1=[]  #define empty row
        for k in xrange(-N,N-1): #total column is 3
            row1.append(jv(i-k,K)*np.exp(-0.5*hbar*1j*i**2)*1j**(i-k)) #adding Bessel function for each column for this row
        matrix.append(row1) #add fully defined column into the row
    matrix = np.array(matrix)
    return LA.matrix_power(matrix, nt)
    

#Preparing the p-matrix
x = np.array(range(-N,N-1))
pmat = hbar*np.diag(x)



#Preparing the theta-matrix
theta=[]

for i in xrange(-N,N-1): #total row is 3
    row2=[]  #define empty row
    for k in xrange(-N,N-1): #total column is 3
        if i==k:
            row2.append(pi)
        else:
            row2.append(1j/(i-k)) 
    theta.append(row2) #add fully defined column into the row
theta = np.array(theta)


#Calculating OTOC with theta(t) and p(0)
def mat1(nt,K):
    return np.matmul(theta,matrixN(nt,K))

def mat2(nt,K):
    return np.matmul(matrixN(nt,K).conj().T,mat1(nt,K))

def mat3(nt,K):
    return np.matmul(mat2(nt,K), pmat)

def mat4(nt,K):
    return np.matmul(pmat,mat2(nt,K))

def mat5(nt,K):
    return LA.matrix_power((mat3(nt,K)-mat4(nt,K)), 2)


#Calculating OTOC with p(t) and p(0)
def mat11(nt,K):
    return np.matmul(pmat,matrixN(nt,K))

def mat22(nt,K): 
    return np.matmul(matrixN(nt,K).conj().T,mat11(nt,K))

def mat33(nt,K):
    return np.matmul(mat22(nt,K), pmat)

def mat44(nt,K):
    return np.matmul(pmat,mat22(nt,K))

def mat55(nt,K):
    return LA.matrix_power((mat33(nt,K)-mat44(nt,K)), 2)


#Initial state
ini=[]
row3=[]

for i in xrange(-N,N-1):
    #row=[]
    row3.append(np.exp(-A*(i-n0)**2))
ini.append(row3)
ini = np.array(ini)
ini = ini.transpose()

denom = np.dot(ini.conj().T, ini)
#print denom
#print ini

#expectation value of the commutator with theta(t) and p(0), on the initial state
def mat6(nt,K):
    return np.matmul(mat5(nt,K),ini)

def mat_res(nt,K):
    return -np.dot(ini.conj().T,mat6(nt,K))/denom**2


#expectation value of the commutator with p(t) and p(0), on the initial state
def mat66(nt,K):
    return np.matmul(mat55(nt,K),ini)

def mat_res_p(nt,K):
    return -np.dot(ini.conj().T,mat66(nt,K))/denom**2




#Plotting OTOC
mat_resp_vec = np.vectorize(mat_res_p)
inp = np.array(range(1,t))
#out1 = mat_resp_vec(inp,0.5)
#out2 = mat_resp_vec(inp,1)
#out3 = mat_resp_vec(inp,2)
out4 = mat_resp_vec(inp,4)
#out5 = mat_resp_vec(inp,6)
#out6 = mat_resp_vec(inp,10)


import matplotlib.pyplot as plt
# plt.semilogy(inp, out1,'r', inp, out2,'b', inp, out3,'g', inp, out4, 'y', inp, out5,'r--', inp, out6, 'b--')
#plt.semilogy(inp, out1, 'r', inp, out3,'g', inp, out5, 'b')
plt.semilogy(inp, out4)
plt.xlabel('time')
plt.ylabel('OTOC')


plt.show()