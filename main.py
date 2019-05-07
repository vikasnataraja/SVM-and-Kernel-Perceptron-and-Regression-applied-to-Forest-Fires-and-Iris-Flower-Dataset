#!/usr/bin/env python
# coding: utf-8

# # SVM and Regularization
# 
# 

# ## Part 1: Lasso and Ridge regression 

# The difference between LASSO and Ridge regression is due to their different regularization model (L1 norm vs. L2 norm). The regularization term in LASSO is $\lambda||\mathbf w||_1$, while the regularization term in ridge regression is  $(\lambda/2) ||\mathbf w||^2$ (where $\mathbf w$ denotes the set of parameters for the linear regression model and $\lambda$ is the trade-off regularization parameter). LASSO typically enforces more _sparsity_ on the resulting $\mathbf w$. That is, the resulting classifier will have a small number of non-zero weights. 

# ### Datasets

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
class DataA:
    def __init__(self):
        f = lambda x, y : np.random.randn(x, y)
        self.train_x = f(1000, 20)
        self.train_y = f(1000, 1)[:,0]
        self.test_x = f(500, 20)
        self.test_y = f(500, 1)[:,0]
        
class DataB:
    def __init__(self):
        # Data from: https://archive.ics.uci.edu/ml/datasets/Cloud
        data = np.fromfile("data/cloud.data", sep = " ").reshape((1024, 10))
        y = data[:, 6]
        X = np.delete(data, 6, axis = 1)
        
        self.train_x = X[:800]
        self.train_y = y[:800]
        
        self.test_x = X[800:]
        self.test_y = y[800:]
        
class DataC:
    def __init__(self):
        # Data from: http://archive.ics.uci.edu/ml/datasets/Forest+Fires
        data = pd.read_csv("data/forestfires.csv")
        data = data.sample(frac = 1).reset_index(drop = True).drop(columns = ["month", "day"])
        data["area"] = np.log(data["area"] + 1)
        X = data.drop(columns = "area").values
        y = data["area"].values
        
        self.train_x = X[:400]
        self.train_y = y[:400]
        
        self.test_x = X[400:]
        self.test_y = y[400:]


# In[ ]:


data_a = DataA()
data_b = DataB()
data_c = DataC()
#print(np.shape(data_a.train_x[:,0]))
#plt.scatter(data_a.train_x[:,0],data_a.train_y)
#plt.title('Data A with X[0]')


# 
# 1. Train a LASSO model using 5 different values for the regularization parameter $\lambda$.
# 2. Report the value of $\lambda$ that yields the minimum number of non-zero coefficients in the resulting $\mathbf w$, and report the number of non-zero coefficients in that case. 
# 3. For each of the classifiers learned in part (A), compute their test error as mean-squared-error. Plot the test error as function of $\lambda$.
# 4. Report the value of $\lambda$ that yields the $\mathbf w$ with the minimum test error. Save this $\mathbf w$ as $\mathbf w_d$. 
# 
# Note: $\lambda$ is same as $\alpha$ in the sklearn module.

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

coeffs_dataA = []
coeffs_dataB = []
coeffs_dataC = []
predA = []
predB = []
predC = []

# alpha values are also called regularization Parameters

regularizationParam = [0.01, 0.05, 0.1, 0.2, 0.3]
#regularizationParam['alpha']

"""
 Construct Lasso models for each dataset.
 Once the model is trained on training set, use predict method to predict values of y.
 Next, calculate Mean-Squared Errors and also get the count of non-zero coeeficients.
"""


for value in regularizationParam:
    #print(values[i])
    lassoRegressiondataA = Lasso(alpha = value)
    lassoModeldataA = lassoRegressiondataA.fit(data_a.train_x, data_a.train_y)
    lassoRegressiondataB = Lasso(alpha = value)
    lassoModeldataB = lassoRegressiondataB.fit(data_b.train_x, data_b.train_y)
    lassoRegressiondataC = Lasso(alpha = value)
    lassoModeldataC = lassoRegressiondataC.fit(data_c.train_x, data_c.train_y)
    coeffs_dataA.append(np.count_nonzero(lassoModeldataA.coef_))
    predA.append(lassoRegressiondataA.predict(data_a.test_x))
    coeffs_dataB.append(np.count_nonzero(lassoModeldataB.coef_))
    predB.append(lassoRegressiondataB.predict(data_b.test_x))
    coeffs_dataC.append(np.count_nonzero(lassoModeldataC.coef_))
    predC.append(lassoRegressiondataC.predict(data_c.test_x))
    
    """
    The following print statements print out the number of non-zero coefficients for each value of alpha
    for each dataset.

    
    """
    
    print('Number of non-zero coefficients (datset A) for ALPHA =',value,' is ',np.count_nonzero(lassoModeldataA.coef_), 'out of ',len(lassoModeldataA.coef_))
    print('Number of non-zero coefficients (dataset B) for ALPHA =',value,' is ',np.count_nonzero(lassoModeldataB.coef_), 'out of ',len(lassoModeldataB.coef_))
    print('Number of non-zero coefficients (dataset C) for ALPHA =',value,' is ',np.count_nonzero(lassoModeldataC.coef_), 'out of ',len(lassoModeldataC.coef_))


# ### Plot the figures ###
# 
# * First figure plots the number of non-zero coefficients for each dataset for each value of $\lambda$
# 
# * Second figure plots the mean squared error for each value of $\lambda$
# 

# In[ ]:


plt.figure()
ax = plt.gca()
ax.scatter(regularizationParam, coeffs_dataA)
ax.scatter(regularizationParam, coeffs_dataB)
ax.scatter(regularizationParam, coeffs_dataC)

ax.plot(regularizationParam, coeffs_dataA,label = 'Dataset A')
ax.plot(regularizationParam, coeffs_dataB,label = 'Dataset B')
ax.plot(regularizationParam, coeffs_dataC, label = 'Dataset C')
plt.title('Alpha vs number of non-zero coefficients for Lasso Regression Model')
plt.xlabel('Alpha (Regularization parameter)')
plt.ylabel('Number of non-zero coefficients')
plt.legend(loc = 'best')

MSE_A = []
MSE_B = []
MSE_C = []
 
for count in range(len(regularizationParam)):
    MSE_A.append(mean_squared_error(data_a.test_y,predA[count]))
    MSE_B.append(mean_squared_error(data_b.test_y,predB[count]))
    MSE_C.append(mean_squared_error(data_c.test_y,predC[count]))
    
plt.figure()
ax = plt.gca()
ax.scatter(regularizationParam,MSE_A)
ax.scatter(regularizationParam,MSE_B)
ax.scatter(regularizationParam,MSE_C)
ax.plot(regularizationParam,MSE_A, label = 'Dataset A')
ax.plot(regularizationParam,MSE_B, label = 'Dataset B')
ax.plot(regularizationParam,MSE_C, label = 'Dataset C')
plt.xlabel('Alpha (Regularization parameter)')
plt.ylabel('Mean-Squared Error')
plt.title('lambda (alpha) vs Mean Squared Error for Lasso Regression')
plt.legend(loc = 'best')

print('MSE of dataset A = ',MSE_A)
print('MSE of dataset B = ',MSE_B)
print('MSE of dataset C = ',MSE_C)


# In[ ]:


"""
I am recording the values of lambda using dictionaries instead of tables
because it is easier to access the items in a dictionary
"""


LassodictCoeffsA = dict(zip(regularizationParam,coeffs_dataA))
LassodictCoeffsB = dict(zip(regularizationParam,coeffs_dataB))
LassodictCoeffsC = dict(zip(regularizationParam,coeffs_dataC))

LassodictErrorsA = dict(zip(regularizationParam,MSE_A))
LassodictErrorsB = dict(zip(regularizationParam,MSE_B))
LassodictErrorsC = dict(zip(regularizationParam,MSE_C))

wb_A_Lasso = min(LassodictCoeffsA.items(), key=lambda x: x[1])
wb_B_Lasso = min(LassodictCoeffsB.items(), key=lambda x: x[1])
wb_C_Lasso = min(LassodictCoeffsC.items(), key=lambda x: x[1])

wd_A_Lasso = min(LassodictErrorsA.items(), key=lambda x: x[1])
wd_B_Lasso = min(LassodictErrorsB.items(), key=lambda x: x[1])
wd_C_Lasso = min(LassodictErrorsC.items(), key=lambda x: x[1])


# ### Observations from Part 1: Lasso Regression Model###
# 
# 1A) 
# * Number of non-zero coefficients are reported above in print statements for each dataset and for each $\lambda$ value.
# 
# 1B) The value of Lambda that yields the minimum number of non-zero coefficients:
# * For dataset A, $\lambda$ = 0.1, 0.2, 0.3 all yielded zero number of non-zero coefficients (meaning all were zero).
# * For dataset B, $\lambda$ = 0.3 yielded 6 non-zero coefficients.
# * For dataset C, $\lambda$ = 0.3 yielded 3 non-zero coefficients.
# 
# 1C) 
# * The Mean-Squared Errors are plotted above (second figure). The values are also printed above.
# 
# 1D) 
# * From the final 3 figures, the value of $\lambda$ with least MSE is:
#     * For dataset A, $\lambda$ = 0.05 yielded the $\mathbf w$ with the least MSE.
#     * For dataset B, $\lambda$ = 0.01 yielded the $\mathbf w$ with the least MSE.
#     * For dataset C, $\lambda$ = 0.1 yielded the $\mathbf w$ with the least MSE.
# 
# 1E)
# * For dataset A, $\lambda$ = 0.1, 0.2, 0.3 all already yielded $\mathbf w$ = 0.
#     * This shows that for dataset A, increasing $\lambda$ does indeed reduce the value of $\mathbf w$ to zero.

# 2. Now, Ridge regression. I'll use sklearn module `Ridge` (read more [here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)). Repeat each of the experiments above using Ridge regression. Use $\lambda = 1, 50, 100, 200, 1000$.

# In[ ]:


from sklearn.linear_model import Ridge

ridgecoeffs_dataA = []
ridgecoeffs_dataB = []
ridgecoeffs_dataC = []
ridgepredA = []
ridgepredB = []
ridgepredC = []
ridgeRegularizationParam = [1, 50, 100, 200, 1000]
#regularizationParam['alpha']

"""
 Construct Ridge models for each dataset
 Once the model is trained on training set, use predict method to predict values of y
 Next, calculate Mean-Squared Errors and also get the count of non-zero coeeficients
"""

for value in ridgeRegularizationParam:
    
    ridgeRegressiondataA = Ridge(alpha=value)
    ridgeModeldataA = ridgeRegressiondataA.fit(data_a.train_x, data_a.train_y)
    ridgeRegressiondataB = Ridge(alpha = value)
    ridgeModeldataB = ridgeRegressiondataB.fit(data_b.train_x, data_b.train_y)
    ridgeRegressiondataC = Ridge(alpha = value)
    ridgeModeldataC = ridgeRegressiondataC.fit(data_c.train_x, data_c.train_y)
    ridgecoeffs_dataA.append(np.count_nonzero(ridgeModeldataA.coef_))
    ridgepredA.append(ridgeRegressiondataA.predict(data_a.test_x))
    ridgecoeffs_dataB.append(np.count_nonzero(ridgeModeldataB.coef_))
    ridgepredB.append(ridgeRegressiondataB.predict(data_b.test_x))
    ridgecoeffs_dataC.append(np.count_nonzero(ridgeModeldataC.coef_))
    ridgepredC.append(ridgeRegressiondataC.predict(data_c.test_x))
    
    """
    The following print statements print out the number of non-zero coefficients for each value of alpha
    for each dataset.
    They are in reference to first part of question 1A
    
    """
    
    print('Number of non-zero coefficients (datset A) for ALPHA =',value,' is ',np.count_nonzero(ridgeModeldataA.coef_), 'out of ',len(ridgeModeldataA.coef_))
    print('Number of non-zero coefficients (dataset B) for ALPHA =',value,' is ',np.count_nonzero(ridgeModeldataB.coef_), 'out of ',len(ridgeModeldataB.coef_))
    print('Number of non-zero coefficients (dataset C) for ALPHA =',value,' is ',np.count_nonzero(ridgeModeldataC.coef_), 'out of ',len(ridgeModeldataC.coef_))

    


# ### Plot the figures ###
# 
# * First figure plots the number of non-zero coefficients for each dataset for each value of $\lambda$
# 
# * Second figure plots the mean squared error for each value of $\lambda$
# 

# In[ ]:


plt.figure()
ax = plt.gca()
ax.scatter(ridgeRegularizationParam, ridgecoeffs_dataA)
ax.scatter(ridgeRegularizationParam, ridgecoeffs_dataB)
ax.scatter(ridgeRegularizationParam, ridgecoeffs_dataC)

ax.plot(ridgeRegularizationParam, ridgecoeffs_dataA,label = 'Dataset A')
ax.plot(ridgeRegularizationParam, ridgecoeffs_dataB,label = 'Dataset B')
ax.plot(ridgeRegularizationParam, ridgecoeffs_dataC, label = 'Dataset C')
plt.title('Alpha vs number of non-zero coefficients for Ridge Regression Model')
plt.xlabel('Alpha (Regularization parameter)')
plt.ylabel('Number of non-zero coefficients')
plt.legend(loc = 'best')

Ridge_MSE_A = []
Ridge_MSE_B = []
Ridge_MSE_C = []
 
for count in range(len(ridgeRegularizationParam)):
    Ridge_MSE_A.append(mean_squared_error(data_a.test_y,ridgepredA[count]))
    Ridge_MSE_B.append(mean_squared_error(data_b.test_y,ridgepredB[count]))
    Ridge_MSE_C.append(mean_squared_error(data_c.test_y,ridgepredC[count]))
    
plt.figure()
ax = plt.gca()
ax.scatter(ridgeRegularizationParam,Ridge_MSE_A)
ax.scatter(ridgeRegularizationParam,Ridge_MSE_B)
ax.scatter(ridgeRegularizationParam,Ridge_MSE_C)
ax.plot(ridgeRegularizationParam,Ridge_MSE_A, label = 'Dataset A')
ax.plot(ridgeRegularizationParam,Ridge_MSE_B, label = 'Dataset B')
ax.plot(ridgeRegularizationParam,Ridge_MSE_C, label = 'Dataset C')
plt.xlabel('Alpha (Regularization parameter)')
plt.ylabel('Mean-Squared Error')
plt.title('lambda (alpha) vs Mean Squared Error for Ridge Regression Model')
plt.legend(loc = 'best')

print('MSE of dataset A = ',Ridge_MSE_A)
print('MSE of dataset B = ',Ridge_MSE_B)
print('MSE of dataset C = ',Ridge_MSE_C)


# In[ ]:


"""
I am recording the values of lambda using dictionaries instead of tables
because it is easier to access the items in a dictionary
"""


RidgedictCoeffsA = dict(zip(ridgeRegularizationParam,ridgecoeffs_dataA))
RidgedictCoeffsB = dict(zip(ridgeRegularizationParam,ridgecoeffs_dataB))
RidgedictCoeffsC = dict(zip(ridgeRegularizationParam,ridgecoeffs_dataC))

RidgedictErrorsA = dict(zip(ridgeRegularizationParam,Ridge_MSE_A))
RidgedictErrorsB = dict(zip(ridgeRegularizationParam,Ridge_MSE_B))
RidgedictErrorsC = dict(zip(ridgeRegularizationParam,Ridge_MSE_C))

wb_A_Ridge = min(RidgedictCoeffsA.items(), key=lambda x: x[1])
wb_B_Ridge = min(RidgedictCoeffsB.items(), key=lambda x: x[1])
wb_C_Ridge = min(RidgedictCoeffsC.items(), key=lambda x: x[1])

wd_A_Ridge = min(RidgedictErrorsA.items(), key=lambda x: x[1])
wd_B_Ridge = min(RidgedictErrorsB.items(), key=lambda x: x[1])
wd_C_Ridge = min(RidgedictErrorsC.items(), key=lambda x: x[1])


# ### Observations from Part 1: Ridge Regression Model###
# 
# 1A) 
# * Number of non-zero coefficients are reported above in print statements for each dataset and for each $\lambda$ value.
# 
# 
# 1B) The value of Lambda that yields the minimum number of non-zero coefficients:
# * For dataset A, no value of $\lambda$ yielded minimum number of non-zero coefficients (meaning all coefficients were non-zero or no coefficient was zero). 
#     * We can just take the lowest value instead for calcualtion. So take $\lambda$ = 1
# *  For dataset A, no value of $\lambda$ yielded minimum number of non-zero coefficients (meaning all coefficients were non-zero or no coefficient was zero). 
#     * We can just take the lowest value instead for calcualtion. So take $\lambda$ = 1
# * For dataset A, no value of $\lambda$ yielded minimum number of non-zero coefficients (meaning all coefficients were non-zero or no coefficient was zero). 
#     * We can just take the lowest value instead for calcualtion. So take $\lambda$ = 1
#     
# 1C) 
# * The Mean-Squared Errors are plotted above (second figure). The values are also printed above.
# 
# 1D) 
# * From the final 3 figures, the value of $\lambda$ with least MSE is:
#     * For dataset A, $\lambda$ = 1000 yielded the $\mathbf w$ with the least MSE.
#     * For dataset B, $\lambda$ = 1 yielded the $\mathbf w$ with the least MSE.
#     * For dataset C, $\lambda$ = 1000 yielded the $\mathbf w$ with the least MSE.
# 
# 1E)
# * For dataset A, increasing the value of $\lambda$ to 100,000 still did not change the number of non-zero coefficients (shown below).
#     * So even if value of $\lambda$ is increased beyond normal values, outcome does not change in the context of number of non-zero coefficients

# In[ ]:


Ridgequestion1E = Ridge(alpha=100000)
ridgetrain1E = Ridgequestion1E.fit(data_a.train_x, data_a.train_y)
print('With alpha =100000','number of non-zero coefficients is', np.count_nonzero(ridgetrain1E.coef_),'out of',len(ridgetrain1E.coef_))


# 3. Compare the two algorithms on each data set: compare the number of non-zero coordinates of the $\mathbf w_d$'s, and compare the test error rates of the $\mathbf w_b$'s.

# In[ ]:


print('\tComparison of wb (number of non-zero coeffs)')
print('\t--------------------------------------------\n')
print('\t Dataset A\t','Dataset B\t','Dataset C')
print('Lasso\t',wb_A_Lasso,'\t',wb_B_Lasso,'\t',wb_C_Lasso)
print('Ridge\t',wb_A_Ridge,'\t',wb_B_Ridge,'\t',wb_C_Ridge)
print('\n\n')
print('\tComparison of wd (test error rates or MSE)')
print('\t--------------------------------------------\n')
print('\t Dataset A\t','Dataset B\t','Dataset C')
print('Lasso\t',wd_A_Lasso,'\t',wd_B_Lasso,'\t',wd_C_Lasso)
print('Ridge\t',wd_A_Ridge,'\t',wd_B_Ridge,'\t',wd_C_Ridge)


# From the above two table comparisons, I can make the following comments:
# 
# * In regards to wb i.e. in the context of least number of non-zero coefficients, Lasso has the least in all three datasets. 
#     * This is particularly because Ridge smooths more with increase in alpha. As alpha increases, the fit becomes more and more strict.
#     * Only if the alpha value is very small, the coefficients vary a lot. As alpha increases, the coefficients become more defined and the fit of the curve becomes strict and tight.
# 
# * In the context of wd i.e. the least error rate, Ridge and Lasso exhibit very similar error rates and are nearly equal.
#     * Dataset B gets the least error rate in the order of 10^-2. Error rates in dataset A and B are nearly equal for both regression models.
#     * It must be noted that while both dataset A and C experienced min. error rate at $\lambda$ = 1000, dataset B experiences the min. error rate at $\lambda$ = 1.

# ## Part 2: Visualizing Data and Decision Boundaries for different kernels 

# ### A. Training SVM with Linear Kernel (Dataset 1)

# In[ ]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from plot_data import plot_data


# Load from data1
mat_data = sio.loadmat('./data/data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()

# Plot training data
plt.figure()
plot_data(X, y)
plt.xlim([0, 4.5])
plt.ylim([1.5, 5])
plt.title("Dataset 1")
plt.show()


# **A:** In `data1.mat`, most of the positive and negative examples can be separated by a single line. Train SVM with linear kernel with C = 1 and plot the decision boundary using `visualize_boundary_linear(X, y, clf)`. `clf` is the SVM classifier. For the classifier, I will use the scikit-learn implementation (Feel free to experiment with different values of C and see what effect it has on the decision boundary)

# In[ ]:



from sklearn import svm
from visualize_boundary_linear import visualize_boundary_linear


# 1. Create a linear SVM classifier
clf = svm.LinearSVC(C=1)

# 2. Fit the model according to the given training data.
clf.fit(X,y)

# 3. Print the mean accuracy on the given train data and labels using the score function in scikit-learn
print('Score for dataset 1 with linear boundary is',clf.score(X,y),'or',clf.score(X,y)*100,'%')



plt.figure()
# Uncomment the below line after you build your classifier
visualize_boundary_linear(X, y, clf)
plt.xlim([0, 4.5])
plt.ylim([1.5, 5])
plt.title("Dataset 1")
plt.show()


# ### B. Training SVM with RBF Kernel (Dataset 2)

# In[ ]:


# Load from data2
mat_data = sio.loadmat('./data/data2.mat')
X = mat_data['X']
y = mat_data['y'].ravel()

# Plot training data
plt.figure()
plot_data(X, y)
plt.xlim([0, 1])
plt.ylim([0.4, 1])
plt.title("Dataset 2")
plt.show()


# **B** In `data2.mat`, the positive and negative data points are not linearly separable. For this dataset, construct an SVM classifier with a Gaussian kernel to learn a non-linear decision boundary. I use the scikit-learn implementation for the same. To plot the decision boundary use `visualize_boundary(X, y, clf)`

# In[ ]:


from visualize_boundary import visualize_boundary

# SVM Parameters to be used
C = 100
gamma = 10

### START CODE HERE ### (approx. 3 lines)
# 1. Create a SVM classifier with kernel='rbf'
clf = svm.SVC(kernel = 'rbf',gamma=gamma, C=C)

# 2. Fit the model according to the given training data.
clf.fit(X,y)

# 3. Print the mean accuracy on the given train data and labels using the score function in scikit-learn
print('Score for dataset 2 with non-linear boundary is ', clf.score(X,y),'or', clf.score(X,y)*100,'%')
### END CODE HERE ### 

plt.figure()
# Uncomment the below line after you build your classifier
visualize_boundary(X, y, clf)
plt.xlim([0, 1])
plt.ylim([0.4, 1])
plt.title("Dataset 2")
plt.show()


# ## Part 3: Kernel Perceptron 

# Here, I will be implementing the Kernel Perceptron using different kernel functions. I will then use the training and test data provided below to train and test the implementation. Finally i will have to report the error rate as a percentage for each kernel function.

# **A:** Implement the kernel functions below:

# In[ ]:


import numpy as np
from numpy import linalg


"""
Linear kernel = transpose(x)*y

Polynomial kernel = (1 + trasnpose(x)*y)^p

Gaussian kernel = exp((-(x-y)^2)/2*(sigma^2))


"""

def linear_kernel(x1, x2):
    #  TODO
    #pass
    return np.dot(x1,x2)

def polynomial_kernel(x, y, p = 3):
    #  TODO
    #  p = degree of the polynomial
    #pass
    return (1 + np.dot(x,y))**p

def gaussian_kernel(x, y, sigma = 0.5):
    #     TODO:
    #pass
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
    


# In[ ]:


import unittest

class TestKernels(unittest.TestCase):
    
    def setUp(self):
        self.x1 = np.array([1, 2, 1])
        self.x2 = np.array([0, 4, -1])
        
    def test0(self):
        """
        Test the linear kernel
        """
        self.assertEqual(linear_kernel(self.x1, self.x2), 7)
    
    def test_polynomial_kernel(self):
        """
        Test the polynomial kernel
        """
        self.assertEqual(polynomial_kernel(self.x1, self.x2), 512)
        
    def test_gaussian_kernel(self):
        """
        Test the gaussian kernel
        """
        self.assertAlmostEqual(gaussian_kernel(self.x1, self.x2) * 10 ** 8, 1.52299, 4)
    

tests = TestKernels()
tests_to_run = unittest.TestLoader().loadTestsFromModule(tests)
unittest.TextTestRunner().run(tests_to_run)
            


# ### Build a class

# In[ ]:


class KernelPerceptron:
    def __init__(self, kernel = linear_kernel, Niter = 1):
        self.kernel = kernel
        self.Niter = Niter
        self.support_vector_x = None
        self.support_vector_y = None
        
    def fit(self, X, y):
        # TODO: 
        #pass
        
        training_examples, training_features = X.shape
        #print('training examples=',training_examples,'also training features=',training_features)
        #print('type of kernel',self.kernel)
        
        # initialize alpha to zero
        self.alpha = np.zeros(training_examples, dtype=np.float64)
        
        # Took a hint from the Piazza discussion and came up with this
        K = np.zeros((training_examples, training_examples))
        #print('Size of k is',K.shape)
        for i in range(training_examples):
            for j in range(training_examples):
                #print('i=',i,'j=',j)
                K[i,j] = self.kernel(X[i], X[j])
                #print('Kij=',K[i,j])
        
        # now update alpha
        # if the y value matches, do nothing
        # if it doesn't match, increment by 1
        
        # again, took a hint from Piazza discussions
        for iterations in range(self.Niter):
            for index in range(training_examples):
                if np.sign(np.sum(K[:,index] * self.alpha * y)) != y[index]:
                    self.alpha[index] += 1
        
        
        
        # set the support vector to take all non-zero values of alpha
        
        supportVector = self.alpha > 0.00001 # or just take non-zero 
        #index = np.arrange(len(self.alpha))[supportVector]
        self.alpha = self.alpha[supportVector]
        self.support_vector_x = X[supportVector]
        self.support_vector_y = y[supportVector]
       
        
    
    def predict(self,X):
        # TODO:
        predicted_y = np.zeros(len(X))
        #print('length of X',len(X))
        #print('zip',zip(self.alpha,self.support_vector_y,self.support_vector_x))
        for i in range(len(X)):
            
            # predict function is given by:
            # y = sign(sum(alpha(i)*y(i)*kernel(X(i),x)))
            
            for a,b,c in zip(self.alpha,self.support_vector_y,self.support_vector_x):
                predicted_y[i]+=a*b*self.kernel(X[i],c)
          
        return np.sign(predicted_y)
       
  
        


# ##  Solving a new classification problem using SVM

# We are using the IRIS Dataset for this classification task. We have created a binary classification problem to determine whether a given flower is a setosa or not. To create this, we pre-processed the labels to create a label vector where setosa’s label is unchanged (i.e. its label is 1), but both versicolor and virginica are now labeled as -1. The data contains two out of the four attributes, petal width and petal length.We are going to use this dataset to test our Kernel Perceptron
# <br>
# 
# <img src="wide_iris.png" alt="Drawing" style="width: 600px;"/>
# 
# <br>

# In[ ]:


from scipy.io import loadmat
class Data:
    def __init__(self):
        ff = lambda x,y : loadmat(x)[y]
        
        self.X_train = ff("data/iris_3/train_data.mat", "train_data")
        self.y_train = ff("data/iris_3/train_labels.mat", "train_labels").flatten()
        
        self.X_test = ff("data/iris_3/test_data.mat", "test_data")
        self.y_test = ff("data/iris_3/test_labels.mat", "test_labels").flatten()
        
data = Data()


# ### Linear Kernel ###

# In[ ]:


perceptLinear = KernelPerceptron(Niter=20, kernel=linear_kernel)
perceptLinear.fit(data.X_train,data.y_train)
y_predicted = perceptLinear.predict(data.X_test)

accuracy_count = 0

for i in range(len(data.y_test)):
    if y_predicted[i] == data.y_test[i]:
        accuracy_count +=1
    
print('accuracy score =',accuracy_count*100/len(data.y_test),'%','and error score is',(len(data.y_test)-accuracy_count)*100/len(data.y_test),'%')


# ### Polynomial Kernel ###

# In[ ]:


perceptPolynomial = KernelPerceptron(Niter=20, kernel=polynomial_kernel)
perceptPolynomial.fit(data.X_train,data.y_train)
y_predicted = perceptPolynomial.predict(data.X_test)

accuracy_count = 0

for i in range(len(data.y_test)):
    if y_predicted[i] == data.y_test[i]:
        accuracy_count +=1

print('accuracy score =',accuracy_count*100/len(data.y_test),'%','and error score is',(len(data.y_test)-accuracy_count)*100/len(data.y_test),'%')


# ### Gaussian Kernel ###

# In[ ]:


perceptGaussian = KernelPerceptron(Niter=20, kernel=gaussian_kernel)
perceptGaussian.fit(data.X_train,data.y_train)
y_predicted = perceptGaussian.predict(data.X_test)

accuracy_count = 0

for i in range(len(data.y_test)):
    if y_predicted[i] == data.y_test[i]:
        accuracy_count +=1

print('accuracy score =',accuracy_count*100/len(data.y_test),'%','and error score is',(len(data.y_test)-accuracy_count)*100/len(data.y_test),'%')


# **C:** Report the test error as a percentage for each kernel function for `Niter = 20`

# ### Test error scores ###
# 
# * The perceptron was run with three different kernels (linear, polynomial and Gaussian) for 20 iterations `Niter=20`
# 
# * The test errors match the expected output:
#     1. Linear kernel has an accuracy of 60% and error score of 40%
# 
#     2. Polynomial kernel has an accuracy of 93.33% and error score of 6.67%
# 
#     3. Gaussian kernel has an accuracy of 93.33% and error score of 6.67%

# ### End of document ###
