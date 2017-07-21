import argparse
import tensorflow as tf
import numpy as np
import time
import os
import random
import logging
import data_aug

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dren_z2cnn', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--trained_model', default='snapshot/dren_z2cnn/final/model_final.ckpt', type=str)  
parser.add_argument('--val_data_path', default='data/rmnist/test_data.npy', type=str)
parser.add_argument('--gpu', default=3, type=int)
args = parser.parse_args()
test_aug_params = {
	'crop_size': [28,28,1],
	'min_border': [0,0,0],
	'zoom':1.0,
	'random_crop':False,
	'sharpen': False,
	'blur_sigma': 0,
	'noise_max_amplitude': 0,
	'flip': False,
	'rot': False,
	}
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if args.model=='dren_z2cnn':
	import models.dren_z2cnn as model
elif args.model=='dren_resnet20':
	import models.dren_resnet20 as model
elif args.model=='dren_z2cnn_x4':
	import models.dren_z2cnn_x4 as model
elif args.model=='z2cnn':
	import models.z2cnn as model
else:
	print 'no corresponding model, import z2cnn'
	import models.z2cnn as model

batch_size = args.batch_size # batch_size <=4 , for memory constrain
num_classes = 10
## load data    

validation_data = np.load(args.val_data_path)

crop_size = np.array(test_aug_params['crop_size'])
x = tf.placeholder('float',shape=[batch_size,crop_size[0],crop_size[1],crop_size[2]])
y = tf.placeholder('float')


def get_normalized_data(iters,batch_size,data,aug_params,istrain=1):

    X=[data_aug.data_aug_2d(data[i,0],aug_params).astype(float) for i in range(iters*batch_size,(iters+1)*batch_size)]
    Y=np.zeros([batch_size,num_classes])
    for i in range(iters*batch_size,(iters+1)*batch_size):
        Y[i-iters*batch_size,data[i,1]]=1
    X=np.reshape(X,[-1,crop_size[0],crop_size[1],crop_size[2]])    
    return X,Y

def test_neural_network(x):
    
    # Setting
    prediction=model.inference_small(x,is_training=False,num_classes=10)
    loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_sum(tf.cast(correct, 'float'))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session(config=config) as sess:
        # Init
        if args.pretrain:
            saver.restore(sess,args.pretrain)
            print 'load',args.pretrain
        else:
            print 'no pretrian model!'
   
		## Validation

        accuracy_num=0
        num=validation_data.shape[0]/batch_size
        epoch_loss=0
        for iters in range(0,validation_data.shape[0]/batch_size):                  
			X,Y=get_normalized_data(iters,batch_size,validation_data,test_aug_params,0)
			c, a, pred = sess.run([loss,accuracy,prediction], feed_dict={x: X, y: Y})                   
			epoch_loss += c
			accuracy_num += a/batch_size
        print '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']VALIDATION:Loss:%.4f;accuarcy:%.4f'%(epoch_loss / num,accuracy_num / num)




# Run this locally:

test_neural_network(x)
