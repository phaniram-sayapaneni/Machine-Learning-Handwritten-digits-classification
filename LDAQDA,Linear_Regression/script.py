
# coding: utf-8

# In[1]:

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import math

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    means=np.array([])
    train=np.append(X,y,axis=1)
    labels = set([tuple(a) for a in y])
    index=0
    for i in labels:
        x=np.where(train[:,2]==i)
        classifier=X[x]
        attr_mean= np.array(classifier.mean(axis=0))
        means=np.append(means,attr_mean,axis=0)

    means=np.reshape(means,(5,2))
    means=np.transpose(means)
    covmat=np.cov(np.transpose(X))
                
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    means=np.array([])
    covmats=[]
    train=np.append(X,y,axis=1)
    labels = set([tuple(a) for a in y])
    index=0
    for i in labels:
        x=np.where(train[:,2]==i)
        classifier=X[x]
        attr_mean= np.array(classifier.mean(axis=0))
        means=np.append(means,attr_mean,axis=0)
        attr_covar = np.cov(np.transpose(classifier))
        covmats.insert(index,attr_covar)
        index+=1
    means=np.reshape(means,(5,2))
    means=np.transpose(means)
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    num_classes = means.shape[1]
    temp =[]
    ypred = []
    cov_inv = np.linalg.inv(covmat)

    for i in range(0,Xtest.shape[0]):
        for j in range(0, num_classes):
            t1=Xtest[i,:]-means[:,j]
            t2 = np.transpose(t1)
            dot1 = np.dot(t2,cov_inv)
            mul = np.dot(dot1,t1)
            num = -1 * 0.5 * mul
            result = np.exp(num)
            temp.insert(j,result)
            ind_max=np.argmax(temp)
            if(j==4):
                del temp[:]
         
        ypred.insert(i, ind_max+1)
               
    acc=str(100 * np.mean((ypred == ytest.flatten()).astype(float)))
    ypred= np.asarray(ypred)
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    d = Xtest.shape[1]
    num_classes = means.shape[1]
    temp =[]
    ypred = []
    for i in range(0,Xtest.shape[0]):
        for j in range(0, num_classes):
            t1 = Xtest[i,:]-means[:,j]
            t2= np.transpose(t1)
            dot1 = np.dot(t2,np.linalg.inv(covmats[j]))
            mul = np.dot( dot1,t1)
            num = -1 * 0.5 * mul
            den1 = np.power(np.pi*2, d/2)
            den2 = np.power(np.linalg.det(covmats[j]),0.5)
            den = den1 * den2
            result = np.exp(num)/den
            temp.insert(j,result)
            ind_max=np.argmax(temp)
            if(j==4):
                del temp[:]
        ypred.insert(i, ind_max+1)
    acc = str(100 * np.mean((ypred == ytest.flatten()).astype(float)))
    ypred= np.asarray(ypred)
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    # IMPLEMENT THIS METHOD
    a= np.dot(np.transpose(X),X)
    b= np.linalg.inv(a)
    c= np.dot(np.transpose(X),y)
    w= np.dot(b,c)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    a= np.dot(np.transpose(X),X)
    I = np.identity(a.shape[1])
    lI = lambd*I
    a= a+lI
    b= np.linalg.inv(a)
    c= np.dot(np.transpose(X),y)
    w= np.dot(b,c)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    # IMPLEMENT THIS METHOD
    a = np.transpose(ytest-np.dot(Xtest,w))
    b = ytest-np.dot(Xtest,w)
    mse = np.dot(a,b)/Xtest.shape[0]
   
    
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    dot1_error = np.dot(w.T, w)
    error1 = 0.5 * lambd
    error1 = error1 * dot1_error  
    shape_w = w.shape[0]
    w2_error = w.reshape((shape_w,1))
    dot2_error = np.dot(X,w2_error)
    temp2_error = y-dot2_error
    square_error = np.square(temp2_error) 
    sum_error = np.sum(square_error,  axis=0)
    error2 = 0.5 * sum_error  
    error =  error1 + error2
    error = error.flatten()
    
#     error_grad calcuation
    dot1_errorgrad = np.dot(y.T, X)
    error_grad1 = -1.0*dot1_errorgrad
    error_grad2 = lambd*w
    dot3_errorgrad = np.dot(X.T, X)
    error_grad3 = np.dot(w.T, dot3_errorgrad)
    error_grad = error_grad1+error_grad2+error_grad3
    error_grad = error_grad.flatten()
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (p+1)) 

    # IMPLEMENT THIS METHOD
    # IMPLEMENT THIS METHOD
    l =len(x)
    Xd = np.ones((l, p+1))
    for i in range(l):
        for j in range(1,p+1): 
            Xd[i,j] = (x[i]**j)
    return Xd








# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# # LDA

means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))


# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept for test data'+str(mle))
print('MSE with intercept for test data'+str(mle_i))

#training data


wtrain = learnOLERegression(X,y)
mle_train = testOLERegression(w,X,y)

wtrain_i = learnOLERegression(X_i,y)
mle_i_train = testOLERegression(wtrain_i,X_i,y)
print('MSE without intercept for training data'+str(mle_train))
print('MSE with intercept for training data'+str(mle_i_train))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.show()


# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


#Problem 5

pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()


# In[ ]:



