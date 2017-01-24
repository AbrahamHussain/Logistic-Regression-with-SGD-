# Abraham Hussain 

# imports 
import numpy as np
import math   
import sys 
import random 
import matplotlib.pyplot as plt

# Global Variables 
e = math.e 



# function that imports the data 
def import_data(file_name): 
   data = np.genfromtxt(file_name, delimiter = ',', dtype = float)

   ones = np.ones((len(data), 1))
   y_value = data[:, 0]
   x_value = data[:, 1:14]
   x_value = np.c_[ones, x_value]

   return(x_value, y_value)


# Normalization function for x 
def train_set_normalization(x_value):
    normalized = np.ones((len(x_value), len(x_value[0])))
    for i in range(1, len(x_value)):
        for j in range(1, len(x_value[0])):
            mean = np.mean(x_value[:,j])
            standard_dev = np.std(x_value[:,j])
            normalized[i][j] = (x_value[i][j] - mean) / standard_dev      
    return (normalized)

# Normalization for the testing set
def test_set_normalization(train_data, test_data): 
    temp = np.ones((test_data.shape))
    for i in range(len(test_data)):
        for j in range(1, len(test_data[i])): 
            mean = np.mean(train_data[:,j])
            standard_dev = np.std(train_data[:,j])
            temp[i][j] = (test_data[i][j] - mean) / standard_dev

    return(temp)



# L2 regularized logistic error function 
def l_2_regularized_logistic_error(weight, x_value, y_value):
    transpose_weight = weight.transpose()
    #cycling through all the points to find the unregularized log loss
    summation = 0  
    for i in range(len(x_value)):
        temp = (1 + e ** (-1 * y_value[i] * np.dot(transpose_weight, x_value[i])))
        temp_2 = math.log(temp)
        summation += temp_2 
    

    return(summation)

# gradient computation 
def gradient(x_value, y_value, lamd, weight): 
    i = random.randint(0, len(x_value) - 1)
    transpose_weight = weight.transpose()
    penalty = (2 * lamd * weight) / float(len(x_value))

    log_loss = (y_value[i] * x_value[i]) / (1 + e **(y_value[i] * np.dot(transpose_weight,x_value[i])))

    return(penalty - log_loss)

# weight update function
def weight_update(weight, step, x_value, y_value, lamd):
    grad = gradient(x_value, y_value, lamd, weight)

    return(weight - (step * grad))


# Stochastic Gradient descent function modified from last week 
def sgd(x_value, y_value, step, weight, lamd, length):
    original_loss = l_2_regularized_logistic_error(weight, x_value, y_value)
    epsilon = 0.0001 
    first_loss = 0 
    
    # first epoch 
    for i in range(1000):
        weight = weight_update(weight, step, x_value, y_value, lamd)


    original_loss_reduction = original_loss - first_loss 
    current_loss_reduction = original_loss_reduction

    result = current_loss_reduction / original_loss_reduction
    
    # runs until the result is less than epsilon 
    while(result >= epsilon):
        temp = l_2_regularized_logistic_error(weight, x_value, y_value)
        for i in range(1000):
                weight = weight_update(weight, step, x_value, y_value, lamd)

        temp_2 = l_2_regularized_logistic_error(weight, x_value, y_value)
        current_loss_reduction = temp - temp_2
        result = current_loss_reduction / original_loss_reduction

    final_loss = l_2_regularized_logistic_error(weight, x_value, y_value)

    return(weight, final_loss)



#main function that runs the stochastic gradient descent and plots the point 
def __main__():
    wine_testing = '/Users/AbrahamHussain/Desktop/Set 2/supplementary_materials/data/wine_testing.txt'
    wine_training_1 = '/Users/AbrahamHussain/Desktop/Set 2/supplementary_materials/data/wine_training1.txt'
    wine_training_2 = '/Users/AbrahamHussain/Desktop/Set 2/supplementary_materials/data/wine_training2.txt'

    train_1x = import_data(wine_training_1)[0]
    train_1y = import_data(wine_training_1)[1]

    train_1x_norm = train_set_normalization(train_1x)

    train_2x = import_data(wine_training_2)[0]
    train_2y = import_data(wine_training_2)[1]
    train_2x_norm = train_set_normalization(train_2x)


    test_x = import_data(wine_testing)[0]

    norm_test_x_1 = test_set_normalization(train_1x, test_x)
    norm_test_x_2 = test_set_normalization(train_2x, test_x)

    test_y = import_data(wine_testing)[1]
    
   

    lamd = [] 
    for i in range(15):
        lamd.append(.0001 * 5**(i))

    error_result_1 = np.array([])
    test_result_1 = np.array([])
    norm_result_1 = np.array([]) 


    error_result_2 = np.array([])
    test_result_2 = np.array([])
    norm_result_2 = np.array([]) 
    


    for i in range(len(lamd)):
        weight = [.001] * 14
        weight = np.array(weight)
        sgd_result = sgd(train_1x_norm, train_1y, 10**-6, weight, lamd[i], len(train_1x_norm))
        weight = sgd_result[0]
        temp = sgd_result[1]
        mean = temp / len(train_1x_norm)
        error_result_1 = np.append(error_result_1, mean)
        error = l_2_regularized_logistic_error(weight, norm_test_x_1, test_y) / float(len(norm_test_x_1))
        test_result_1 = np.append(test_result_1, error)
        norm_result_1 = np.append(norm_result_1, np.linalg.norm(weight))

    for j in range(len(lamd)):
        weight = [.001] * 14
        weight = np.array(weight)
        sgd_result = sgd(train_2x_norm, train_2y, 10**-6, weight, lamd[j], len(train_2x_norm))
        weight = sgd_result[0]
        temp = sgd_result[1]
        mean = temp / len(train_2x_norm)
        error_result_2 = np.append(error_result_2, mean)
        error = l_2_regularized_logistic_error(weight, norm_test_x_2, test_y) / float(len(norm_test_x_2))
        test_result_2 = np.append(test_result_2, error)
        norm_result_2 = np.append(norm_result_2, np.linalg.norm(weight))



 



    plt_1 = plt.plot(lamd, error_result_1, label="Testing set 1")
    plt_2 = plt.plot(lamd,error_result_2, label="Testing set 2")


    plt.title("SGD for l2 logistic regression (Ein)") 
    plt.xlabel("Lamda")
    plt.xscale('log')
    plt.ylabel("Error Result") 
    plt.legend()
    plt.show()


    plt_3 = plt.plot(lamd, test_result_1, label = "Testing set 1")
    plt_4 = plt.plot(lamd, test_result_2, label = "Testing set 2")


    plt.title("SGD for l2 logistic regression (Eout)") 
    plt.xlabel("Lamda")
    plt.xscale('log')
    plt.ylabel("Error Result") 
    plt.legend()
    plt.show()


    plt_5 = plt.plot(lamd, norm_result_1, label = "Testing set 1")
    plt_6 = plt.plot(lamd, norm_result_2, label = "Testing set 2")

    plt.title("Norm of w") 
    plt.xlabel("Lamda")
    plt.xscale('log')
    plt.ylabel("Norms") 
    plt.legend()
    plt.show()
        


    
__main__()
    




