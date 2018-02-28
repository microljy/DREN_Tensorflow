import numpy as np
#import matplotlib.pyplot as plt



def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def load_data(folder = './data/cifar10/cifar-10-batches-py/'):


    train_data = np.zeros([50000,32,32,3],dtype=np.uint8)
    train_label = np.zeros([50000],dtype=np.uint8)
    for i in range(5):
        filename = 'data_batch_' + str(i+1)
        filepath = folder + filename
        dic = unpickle(filepath)
        data = dic['data']
        labels = dic['labels']
        data = np.reshape(data,[10000,3,32,32])
        data = data.transpose((0,2,3,1))
        train_data[i*10000:(i+1)*10000]=data
        train_label[i*10000:(i+1)*10000]=labels
    
    #mean = np.mean(train_data.astype(float))
    #std = np.std(train_data.astype(float))
    #train_data = (train_data.astype(float) - mean) / std
    #plt.imshow(train_data[2])   
    
    filename = 'test_batch'
    filepath = folder + filename
    dic = unpickle(filepath)
    data = dic['data']
    test_label = dic['labels']
    data = np.reshape(data,[10000,3,32,32])
    test_data = data.transpose((0,2,3,1))
    #test_data = (test_data.astype(float) - mean) / std
    
    train_collection = []
    test_collection = []
    for i in range(len(train_data)):
        train_collection.append([train_data[i],train_label[i]])
    for i in range(len(test_data)):
        test_collection.append([test_data[i],test_label[i]])
    
    return np.array(train_collection),np.array(test_collection)

if __name__=='__main__':
    train_data,test_data = load_data(folder = './cifar-10-batches-py/')