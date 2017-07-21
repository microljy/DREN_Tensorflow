import scipy.io as sio  
import numpy as np  
import matplotlib.pyplot as plt  
#matlab文件名  
matfn='data.mat'  
data=sio.loadmat(matfn)  
x = data['x']
y = data['y']
x_test = data['x_test']
y_test = data['y_test']
traindata = []
testdata = []
for i in range(x.shape[0]):
    traindata.append([x[i,0],y[i]])

for i in range(x_test.shape[0]):
    testdata.append([x_test[i,0],y_test[i]])

np.save('train_data.npy',traindata)
np.save('test_data.npy',testdata)