from sklearn import linear_model 
import numpy as np 
import matplotlib.pyplot as plt


# import data 
def import_data(file_name): 
   data = np.genfromtxt(file_name, delimiter = '\t', dtype = float)
   x_value = data[:, 0:9]
   y_value = data[:,9]

   x_value = np.array(x_value)
   y_value = np.array(y_value)
  
   return(x_value, y_value)


# plots the lasso plot using the sklearn package 
def lasso_plot(x_value, y_value):
    step = 0.01 
    coef_list = [] 
    for i in np.arange(0.0, 2.0, step):
        clf = linear_model.Lasso(alpha=i)
        clf.fit(x_value, y_value)
        coef_list.append(clf.coef_)
    coef_list_a = np.asarray(coef_list)
    print (len(coef_list_a))
    trans_coef = np.transpose(coef_list_a)
    print (len(trans_coef))
    for j in range(9):
        plt.plot(np.arange(0, 2, 0.01), coef_list_a)
    plt.show()

# plots the ridge plot using sklearn 
def ridge_plot(x_value, y_value):
    step = 10 
    coef_list = [] 
    for i in np.arange(0.0, 2000, step):
        clf = linear_model.Ridge(alpha=i)
        clf.fit(x_value, y_value)
        coef_list.append(clf.coef_)
    coef_list_a = np.asarray(coef_list)
    print (len(coef_list_a))
    trans_coef = np.transpose(coef_list_a)
    print (len(trans_coef))
    for j in range(9):
        plt.plot(np.arange(0, 2, 0.01), coef_list_a)
    plt.show()


def __main__():
    filename = '/Users/AbrahamHussain/Desktop/Set 2/supplementary_materials/data/problem3data.txt'
    data  = import_data(filename)

    #print(data[0][0])

    #lasso_plot(data[0], data[1])
    ridge_plot(data[0], data[1])





__main__()