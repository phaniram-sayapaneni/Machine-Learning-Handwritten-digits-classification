
# coding: utf-8

# In[2]:

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0
    

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################

    #print('initialWeights' , initialWeights.shape)
    #print('n_feature+1', n_feature+1)
    #print('train_data.shape', train_data.shape)
    ##
    # initialWeights = initialWeights.reshape((n_feature+1,1))
    initialWeights = np.array([initialWeights]).T
    new_col = np.ones((n_data, 1))
    #################
    #new_col = 0.5*new_col
    #####################
    train_data = np.concatenate((new_col, train_data),1)
    #print('train_data.shape', train_data.shape)
    py = sigmoid(np.dot(train_data, initialWeights))
    #print('py.shape', py.shape)
    #py.reshape((n_data,1))
    #print('py.shape', py.shape)
    onepy = 1.0 -py
    #print('onepy.shape', onepy.shape)
    Logpy = np.log(py)
    #Logpy = np.array([Logpy]).T
    #print('Logpy.shape', Logpy.shape)
    Log1py = np.log(onepy)
    #Log1py =np.array([Log1py]).T
    #print('Log1py.shape', Log1py.shape)
    #print('Labeli.shape', labeli.shape)
    #main here:
    ##ans  = np.multiply(labeli, Logpy) + np.multiply((1.0 - labeli), Log1py)
    ans = (labeli * Logpy) + ((1.0 - labeli) * Log1py)
    #print('ans', ans)
    error  = (-1.0)*np.sum(ans)
    error = error/n_data
    #print('error1', error)
    #error = error *2.0
    #print('error1', error)
    #print('training data.shape', (train_data.T).shape)
    #print('label data.shape', (np.array([py]).T).shape)

    error_grad = np.dot(train_data.T,(py-labeli))
    error_grad = error_grad.flatten()
    error_grad = (1.0/n_data)*error_grad

    #error_grad= (-1.0)*error_grad

    #print('grad', error_grad.shape)
    #error_grad = error_grad.flatten()


    return error, error_grad
def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    #print('W', W.shape)
    #print('data', data.shape)
    #new_col = np.zeros((data.shape[0], 1)) change it to ones
    new_col = np.ones((data.shape[0], 1))
    data = np.concatenate((new_col, data), 1)
    #print('data', data.shape)
    predict  = sigmoid(np.dot(data, W))
    data = np.concatenate((new_col, data), 1)
    #print('predict', predict.shape)
    label = np.argmax(predict, axis =1)
    #print('label', label.shape)
    label = np.array([label]).T
    #print('label', label.shape)

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data = args[0]
    labeli = args[1]
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    n_feature_b = n_feature + 1
    error_grad = np.zeros((n_feature_b, n_class))
    
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
#     preparing data from input parameters
    num = np.zeros((n_data,n_class))
    theta = np.zeros((n_data,n_class))
    initialWeights = params.reshape(n_feature_b,n_class)

#     adding bias
    input_init_bias = np.ones((n_data,1))
    input_bias = np.concatenate((input_init_bias,train_data), axis = 1)
    
#     calculating the transpose of initial weight and input data
    initialWeights_transpose = np.transpose(initialWeights)
    input_transpose = np.transpose(input_bias)      
    
#     building the numerator
    for i in range(n_class):
        iw_transpose = initialWeights_transpose[i,:]
        num[:,i] = np.dot(iw_transpose,input_transpose)
    num = np.exp(num)
        
#     building the denominator
    den = np.sum(num,axis = 1)
    
#     calculating theta
    for i in range(n_class):
        theta[:,i] = np.divide(num[:,i],den)

#     log of theta
    logtheta = np.log(theta)

#     calculating the error
    labeli_array = np.array(labeli)
    logtheta_array = np.array(logtheta)
    entropy = np.multiply(labeli_array, logtheta_array)
    error = np.sum(entropy)
    error = -1 * error
    
#     calculating the error gradient
    grad_init = theta - labeli
    grad_init = np.transpose(grad_init)
    error_grad = np.dot(grad_init,input_bias)
    error_grad = np.transpose(error_grad)
    error_grad = error_grad.flatten()
    
    return error, error_grad

def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

#     preparing data from input parameters
    n_data = data.shape[0]
    n_feature = data.shape[1]
    n_feature_b = n_feature + 1  
    num = np.zeros((n_data,n_class))
    theta = np.zeros((n_data,n_class))
    initialWeights = W.reshape(n_feature_b,n_class)
    
#     adding bias
    input_init_bias = np.ones((n_data,1))
    input_bias = np.concatenate((input_init_bias,data), axis = 1)
            
#     calculating the transpose of initial weight and input data
    initialWeights_transpose = np.transpose(initialWeights)
    input_transpose = np.transpose(input_bias)     
    
#     building the numerator
    for i in range(n_class):
        iw_transpose = initialWeights_transpose[i,:]
        num[:,i] = np.dot(iw_transpose,input_transpose)
    num = np.exp(num)

            
#     building the denominator
    den = np.sum(num,axis = 1)

#     calculating theta
    for i in range(n_class):
        theta[:,i] = np.divide(num[:,i],den)

    label = np.argmax(theta,axis = 1)  

    label = label.reshape((n_data,1))

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
print('Linear kernel')
clf1= SVC(kernel='linear')
clf1.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf1.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf1.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf1.score(test_data, test_label)) + '%')


# Radial basis function with gamma = 1
print('\n rbf SVM with gamma value = 1')
clf2 = SVC(kernel='rbf', gamma=1.0)
clf2.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf2.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf2.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf2.score(test_data, test_label)) + '%')


# # Radial basis function with gamma = 0
print('\n rbf SVM with gamma value = 0')
clf3 = SVC(kernel='rbf')
clf3.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf3.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf3.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf3.score(test_data, test_label)) + '%')



print ("\n \n C=1")
clf = SVC(C=1,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')


print ("\n \n C=10")
clf = SVC(C=10,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

print ("\n \n C=20")
clf = SVC(C=20,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

print ("\n \n C=30")
clf = SVC(C=30,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

print ("\n \n C=40")
clf = SVC(C=40,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

print ("\n \n C=50")
clf = SVC(C=50,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

print ("\n \n C=60")
clf = SVC(C=60,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

print ("\n\n C=70")
clf = SVC(C=70,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

print ("\n \n C=80")
clf = SVC(C=80,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

print ("\n \n C=90")
clf = SVC(C=90,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

print ("\n \n C=100")
clf = SVC(C=100,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')


# In[ ]:



