
def load_data():
    import numpy as np
    train_path ='./data/rmnist/train_data.npy'
    test_path = './data/rmnist/test_data.npy'
    
    train_data = np.load(train_path)
    validation_data = np.load(test_path)
    return train_data,validation_data
if __name__=='__main__':
    train_data,validation_data = load_data()