# Abraham Hussain 
import numpy as np
import matplotlib.pyplot as plt
import math 
from sklearn.linear_model import LogisticRegression 
from sklearn import svm 


# Imports data from the text file 
def import_data(file_name): 
   data = np.genfromtxt(file_name, delimiter = ',', dtype = float)
   y_value = data[:, 2]
   x_value = data[:,:2]
   
   return(x_value, y_value)

def make_plot(X, y, clf, title, filename):
    '''
    Plots the decision boundary of the classifier <clf> (assumed to have been fitted
    to X via clf.fit()) against the matrix of examples X with corresponding labels y.

    Uses <title> as the title of the plot, saving the plot to <filename>.

    Note that X is expected to be a 2D numpy array of shape (num_samples, num_dims).
    '''
    # Create a mesh of points at which to evaluate our classifier
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8, vmin=-1, vmax=1)

    # Also plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.savefig(filename)
    plt.show()

# logistic regression based on sklearn
def logistic_regression(x_value, y_value):
    logreg = LogisticRegression(class_weight={1:5})

    return (logreg.fit(x_value, y_value))

# svm based on sklearn
def SVM_func(x_value, y_value): 
    clf = svm.SVC(kernel ='linear')
    return(clf.fit(x_value, y_value))

# plots 
def __main__():
    dataset_1 = '/Users/AbrahamHussain/Desktop/Set 2/supplementary_materials/data/problem1dataset1.txt' 
    dataset_2 = '/Users/AbrahamHussain/Desktop/Set 2/supplementary_materials/data/problem1dataset2.txt' 
    dataset_3 = '/Users/AbrahamHussain/Desktop/Set 2/supplementary_materials/data/problem1dataset3.txt'

    data = import_data(dataset_2)
    x_value = data[0]
    y_value = data[1]

    reg_fit = logistic_regression(x_value, y_value)
    svm_results = SVM_func(x_value, y_value)

    make_plot(x_value, y_value, svm_results, "SVM Data Set 2", "SVM2D")

__main__()




