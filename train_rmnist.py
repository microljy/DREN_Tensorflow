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
parser.add_argument('--model', default='z2cnn', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--display', default=300, type=int)
parser.add_argument('--save_every', default=5, type=int)
parser.add_argument('--epoch', default=300, type=int)
parser.add_argument('--decay_epoch', default=200, type=int)
parser.add_argument('--begin_step', default=0, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--shuffle', default=False, type=bool)    
parser.add_argument('--show', default=False, type=bool)
parser.add_argument('--train_data_path', default='data/rmnist/train_data.npy', type=str)
parser.add_argument('--val_data_path', default='data/rmnist/test_data.npy', type=str)
parser.add_argument('--gpu', default=3, type=int)
args = parser.parse_args()
train_aug_params = {
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
elif args.model=='z2cnn':
	import models.z2cnn as model
else:
	print 'no corresponding model'
	import models.res3d20 as model

savePath = './snapshot/save_'+ args.model + '_' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + '/'
if not os.path.exists(savePath):
        os.mkdir(savePath)	
## log	
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%Y %H:%M:%S',
                filename=savePath+'info.log',
                filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-6s: %(levelname)-6s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

## args
logging.info(args)
batch_size = args.batch_size # batch_size <=4 , for memory constrain
display = args.display
hm_epochs = args.epoch
save_every = args.save_every
num_classes = 10
## load data    
logging.info( 'loading data from '+ args.train_data_path)
train_data = np.load(args.train_data_path)
logging.info( 'loading data from '+ args.val_data_path)
validation_data = np.load(args.val_data_path)
if args.shuffle:
	index=range(train_data.shape[0])
	random.shuffle(index)
	train_data=train_data[index]

train_len=(train_data.shape[0]/batch_size)*batch_size# make it times of batch_size
train_data=train_data[0:train_len]    

crop_size = np.array(train_aug_params['crop_size'])
x = tf.placeholder('float',shape=[batch_size,crop_size[0],crop_size[1],crop_size[2]])
y = tf.placeholder('float')


def get_normalized_data(iters,batch_size,data,aug_params,istrain=1):

    X=[data_aug.data_aug_2d(data[i,0],aug_params).astype(float) for i in range(iters*batch_size,(iters+1)*batch_size)]
    Y=np.zeros([batch_size,num_classes])
    for i in range(iters*batch_size,(iters+1)*batch_size):
        Y[i-iters*batch_size,data[i,1]]=1
    X=np.reshape(X,[-1,crop_size[0],crop_size[1],crop_size[2]])    
    return X,Y

def train_neural_network(x):
    
    # Setting
    prediction=model.inference_small(x,is_training=True,num_classes=10)
    loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_sum(tf.cast(correct, 'float'))
        
    global_step = tf.Variable(0, trainable=False)
    num_decay_steps=train_len/batch_size*args.decay_epoch
    learning_rate=tf.train.exponential_decay(args.lr, global_step-args.begin_step, num_decay_steps, 0.1, staircase=True)
    logging.info( 'learning_rate_decay: num_decay_steps:%d'%num_decay_steps)
    base_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    UPDATE_OPS_COLLECTION = 'resnet_update_ops'
    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    optimizer = tf.group(base_optimizer, batchnorm_updates_op)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session(config=config) as sess:
        # Init
        if args.pretrain:
            saver.restore(sess,args.pretrain)
            begin_epoch=global_step.eval()/(train_len/batch_size)
            begin_epoch=0
            print 'load',args.pretrain,'global_step:',global_step.eval(),'begin epoch:',begin_epoch
        else:
            sess.run(tf.global_variables_initializer())
            begin_epoch=0
    
        ## training
        for epoch in range(begin_epoch,hm_epochs):
            epoch_loss = 0
            accuracy_num=0          
            for iters in range(0,train_data.shape[0]/batch_size):
               
                X,Y=get_normalized_data(iters,batch_size,train_data,train_aug_params)                    
                _, c, a, pred = sess.run([optimizer, loss,accuracy,prediction], feed_dict={x: X, y: Y})                   
                epoch_loss += c
                accuracy_num += a/batch_size
						
					## log info
                if (iters+1) % display == 0 or iters == train_data.shape[0]/batch_size -1:
                    logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']TRAINING:Epoch:%d;Display:%d;Loss:%.4f;accuarcy:%.4f;learning_rate:'%(epoch+1, iters / display + 1,epoch_loss / (iters+1),accuracy_num / (iters+1))+str(learning_rate.eval()))

            ## Validation
            accuracy_num=0
            num=validation_data.shape[0]/batch_size
            epoch_loss=0
            for iters in range(0,validation_data.shape[0]/batch_size):                  
                X,Y=get_normalized_data(iters,batch_size,validation_data,test_aug_params,0)
                c, a, pred = sess.run([loss,accuracy,prediction], feed_dict={x: X, y: Y})                   
                epoch_loss += c
                accuracy_num += a/batch_size
            logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']VALIDATION: Epoch:%d;Loss:%.4f;accuarcy:%.4f'%(epoch+1,epoch_loss / num,accuracy_num / num))

            #save
            if (epoch + 1) % args.save_every==0:
                epoch_path=savePath+'epoch%d/'% (epoch+1)
                if not os.path.exists(epoch_path):
                    os.makedirs(epoch_path)
                path=epoch_path+'model.ckpt'
                saver.save(sess,path)
                logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'] save as'+path)
			
        epoch_path=savePath+'final/'
        if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)	
        path=epoch_path+'model_final.ckpt'
        saver.save(sess,path)
        logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'] save as'+path)

# Run this locally:

train_neural_network(x)
