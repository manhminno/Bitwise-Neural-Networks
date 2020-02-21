import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

#Accuracy
def acc(test_label, output_pred):
    correct = np.sum(test_label == output_pred.flatten())
    return correct/test_label.shape[0]

#Precision, Recall, F1score
def pre_re_f1(cfs_matrix):
    p = cfs_matrix[0,0]/np.sum(cfs_matrix[:,0])
    r = cfs_matrix[0,0]/np.sum(cfs_matrix[0])
    f1 = (2*p*r)/(p+r)
    return (p, r, f1)

#Load data
data = pd.read_csv('data.csv')
label = data['7']
data.drop('7', axis=1, inplace=True)
#Chia 80% train, 20% test
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(data, label, test_size=0.2)

#Khoi tao layers, epochs, learning rate
input_L, hid_L1, hid_L2, output_L = 6, 4, 2, 1
epochs = 10000
lr = 0.1

#Khoi tao w, bias cua tung layer 
hid_L1_w = np.random.uniform(size =(input_L, hid_L1))
hid_L1_bias = np.random.uniform(size=(1, hid_L1))

hid_L2_w = np.random.uniform(size=(hid_L1, hid_L2))
hid_L2_bias = np.random.uniform(size=(1, hid_L2))

output_L_w = np.random.uniform(size=(hid_L2, output_L))
output_L_bias = np.random.uniform(size=(1, output_L))

#Fedforward va Backpropagation
for i in range(epochs):
    #Fedforward hidden layers 1
    hid_L1_sum = np.dot(train_inputs.values, hid_L1_w)
    hid_L1_sum += hid_L1_bias
    hid_L1_output = sigmoid(hid_L1_sum)
    #Hidden layer 2
    hid_L2_sum = np.dot(hid_L1_output, hid_L2_w)
    hid_L2_sum += hid_L2_bias
    hid_L2_output = sigmoid(hid_L2_sum)
    #Output layers
    output_L_sum = np.dot(hid_L2_output, output_L_w)
    output_L_sum += output_L_bias
    predict = sigmoid(output_L_sum)

    #Backpropagation tinh d rieng cua tung layer
    #Output layer
    error = np.reshape(train_outputs.values, (-1, 1)) - predict
    d_predict = error * sigmoid_derivative(predict)
    #Hidden2 layer
    error_hid_L2 = d_predict.dot(output_L_w.T)
    d_hid_L2 = error_hid_L2 * sigmoid_derivative(hid_L2_output)
    #Hidden1 layers
    error_hid_L1 = d_hid_L2.dot(hid_L2_w.T)
    d_hid_L1 = error_hid_L1 * sigmoid_derivative(hid_L1_output)

    #Cap nhat lai w va bias
    output_L_w += lr * hid_L2_output.T.dot(d_predict)
    output_L_bias += np.sum(d_predict, axis=0, keepdims=True) * lr
    
    hid_L2_w += lr * hid_L1_output.T.dot(d_hid_L2)
    hid_L2_bias += np.sum(d_hid_L2, axis=0, keepdims=True) * lr
    
    hid_L1_w += lr * train_inputs.T.dot(d_hid_L1)
    hid_L1_bias += np.sum(d_hid_L1, axis=0, keepdims=True) * lr

#Thu predict tren tap test
hid1 = np.dot(test_inputs.values, hid_L1_w) + hid_L1_bias
hid1_output = sigmoid(hid1)

hid2 = np.dot(hid1_output, hid_L2_w) + hid_L2_bias
hid2_output = sigmoid(hid2)

output = np.dot(hid2_output, output_L_w) + output_L_bias
output_pre = sigmoid(output)

#Predict
print('Predict   :', np.reshape((output_pre.round()),(1, -1)).flatten().astype(int))
print('Test_label:', test_outputs.values)

#Acc va F1score
print('Accuracy  : {0:.2f}%'.format(100*acc(test_outputs.values, output_pre.round())))
p, r, f1 = pre_re_f1(confusion_matrix(test_outputs.values, output_pre.round()))
print('Precision : {0:.2f}, Recall: {1:.2f}, F1_Score: {2:.2f}'.format(p, r, f1))