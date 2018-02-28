import argparse
import tensorflow as tf
import numpy as np
import time
import os
import random
import logging
import math
from tensorflow.python.ops import control_flow_ops


## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet20', type=str)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--display', default=1000, type=int)
parser.add_argument('--save_every', default=20, type=int)
parser.add_argument('--val_every', default=2, type=int)
parser.add_argument('--epoch', default=300, type=int)
parser.add_argument('--decay_epoch', default=200, type=int)
parser.add_argument('--begin_step', default=0, type=int)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--optimizer', default='nestrov', type=str)
parser.add_argument('--shuffle', default=False, type=bool)    
parser.add_argument('--show', default=False, type=bool)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--gpu', default=3, type=int)
args = parser.parse_args()
num_classes = 10
if args.dataset == 'rmnist':
    import data.rmnist.load_rmnist as load_data
    channel = 1 
    num_classes = 10
elif args.dataset == 'cifar10':
    import data.cifar10.load_cifar10 as load_data   
    channel = 3 
    num_classes = 10
    
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if args.model=='dren_z2cnn':
	import models.dren_z2cnn as model
elif args.model=='dren_resnet20':
	import models.dren_resnet20 as model
elif args.model=='dren_resnet20_2b':
	import models.dren_resnet20_2b as model
elif args.model=='dren_resnet20_2b_x4':
	import models.dren_resnet20_2b_x4 as model
elif args.model=='dren_z2cnn_x4':
	import models.dren_z2cnn_x4 as model
elif args.model=='mobilenet10':
	import models.mobilenet10 as model
elif args.model=='resnet20':
	import models.resnet20 as model
else:
	print 'no corresponding model, import z2cnn'
	import models.z2cnn as model

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
val_every = args.val_every

## load data    
logging.info( 'loading dataset '+ args.dataset) 
train_data, validation_data = load_data.load_data()



train_len=(train_data.shape[0]/batch_size)*batch_size# make it times of batch_size
train_data=train_data[0:train_len]    

crop_size = np.array([32,32,channel])
x = tf.placeholder('float',shape=[batch_size,crop_size[0],crop_size[1],crop_size[2]])
y = tf.placeholder('float')
is_training = tf.placeholder('bool')
lr=tf.placeholder('float')

def tf_aug(x,is_training,rotate = 0,device_id=0,horizontal_flip=True,vertical_flip=False,zoom_min=1.0,zoom_max=1.0,
           crop_probability=0.8, # How often we do crops
           crop_min_percent=0.875, # Minimum linear dimension of a crop
           crop_max_percent=1.): # Maximum linear dimension of a crop
            
## Reference [Data augmentation on GPU in Tensorflow](https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19)

    x = tf.image.convert_image_dtype(x, dtype=tf.float32)
    
    def zoom(x,zoom):
        x = tf.expand_dims(x, 0)
        #print x
        #height1, width1 = shpa[1], shpa[2]
        height1, width1 = 32,32
        new_height = tf.cast(tf.cast(height1, tf.float32) * zoom,tf.int32)
        new_width = tf.cast(tf.cast(width1, tf.float32) * zoom,tf.int32)
        x = tf.image.resize_bilinear(x, (new_width, new_height))
        x = tf.image.resize_image_with_crop_or_pad(x,height1,width1)
        #print x
        return x

            
        
    with tf.name_scope('augmentation_'+str(device_id)):
        
        transforms = []
        shp = tf.shape(x)
        print x
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
              [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
              tf.where(coin,
                       tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                       tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))
    
        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
              [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
              tf.where(coin,
                       tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                       tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))
      
        if rotate > 0:
          angle_rad = rotate / 180 * math.pi
          angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
          transforms.append(
              tf.contrib.image.angles_to_projective_transforms(
                  angles, height, width))
    
        if crop_probability > 0:
          crop_pct = tf.random_uniform([batch_size], crop_min_percent,crop_max_percent)
          left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
          top = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
          crop_transform = tf.stack([
              crop_pct,
              tf.zeros([batch_size]), top,
              tf.zeros([batch_size]), crop_pct, left,
              tf.zeros([batch_size]),
              tf.zeros([batch_size])
          ], 1)
    
          coin = tf.less(
              tf.random_uniform([batch_size], 0, 1.0), crop_probability)
          transforms.append(
              tf.where(coin, crop_transform,
                       tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))
        if transforms:
            x_aug = tf.contrib.image.transform(
                x,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR') # or 'NEAREST'
            #print x_aug,x  
            
            #if zoom_min<1.0 or zoom_max>1.0:
            #    zooms = tf.random_uniform([batch_size], zoom_min, zoom_max)
            #    x_aug = [zoom(x_aug[i],zooms[i]) for i in range(args.batch_size)]
            #    x_aug = tf.concat(x_aug,0)
            #print x_aug,x  
            x = control_flow_ops.cond(is_training, lambda: x_aug,lambda: x)
            
    return x

    
def get_normalized_data(iters,batch_size,data,istrain=1):

    X=[data[i,0].astype(float) for i in range(iters*batch_size,(iters+1)*batch_size)]
    Y=np.zeros([batch_size,num_classes])
    for i in range(iters*batch_size,(iters+1)*batch_size):
        Y[i-iters*batch_size,data[i,1]]=1
    X=np.reshape(X,[-1,crop_size[0],crop_size[1],crop_size[2]])    
    return X,Y

def train_neural_network(train_data):
    
    # Setting
    UPDATE_OPS_COLLECTION = 'resnet_update_ops'
    CORRELATION_COLLECTION = 'correlation_collection'
    x_ = tf_aug(x,is_training)
    prediction=model.inference_small(x_,is_training,num_classes=num_classes)
    regularization_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cor_losses = 0.0000 * tf.reduce_sum(tf.get_collection(CORRELATION_COLLECTION))
    loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)) + regularization_losses + cor_losses
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_sum(tf.cast(correct, 'float'))
        
    global_step = tf.Variable(0, trainable=False)
    #num_decay_steps=train_len/batch_size*args.decay_epoch
    #learning_rate=tf.train.exponential_decay(args.lr, global_step-args.begin_step, num_decay_steps, 0.1, staircase=True)
    #logging.info( 'learning_rate_decay: num_decay_steps:%d'%num_decay_steps)
    if args.optimizer=='adam':
        base_optimizer = tf.train.AdamOptimizer(lr,epsilon=1e-4).minimize(loss,global_step=global_step)
    elif args.optimizer == 'nestrov':
        base_optimizer = tf.train.MomentumOptimizer(lr,momentum=0.9,use_nesterov=True).minimize(loss,global_step=global_step)
    

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
            step_begin = global_step.eval()
            print 'load',args.pretrain,'global_step:',global_step.eval(),'begin epoch:',begin_epoch
            '''
            learning_rate=tf.train.exponential_decay(args.lr, global_step-step_begin, num_decay_steps, 0.1, staircase=True)
            base_optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9,use_nesterov=True).minimize(loss,global_step=global_step)
            optimizer = tf.group(base_optimizer, batchnorm_updates_op)
            '''
            begin_epoch=0
        else:
            sess.run(tf.global_variables_initializer())
            begin_epoch=0
    
        ## training
        for epoch in range(begin_epoch,hm_epochs):
            if epoch < 160:
                learning_rate = args.lr
            elif epoch < 240:
                learning_rate = args.lr / 10
            else:
                learning_rate = args.lr / 100
                
            if args.shuffle:
                index=range(train_data.shape[0])
                random.shuffle(index)
                train_data=train_data[index]
            epoch_loss = 0
            accuracy_num=0          
            for iters in range(0,train_data.shape[0]/batch_size):
               
                X,Y=get_normalized_data(iters,batch_size,train_data)                    
                _, c, a, pred, cor_loss,r_loss = sess.run([optimizer, loss,accuracy,prediction,cor_losses,regularization_losses], feed_dict={x: X, y: Y, is_training: True,lr: learning_rate})                 
                epoch_loss += c
                accuracy_num += a/batch_size
						
					## log info
                if (iters+1) % display == 0 or iters == train_data.shape[0]/batch_size -1:
                    logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']TRAINING:Epoch:%d;Display:%d;Loss:%.4f;cor_Loss:%.4f;r_Loss:%.4f;accuarcy:%.4f;learning_rate:'%(epoch+1, iters / display + 1,epoch_loss / (iters+1),cor_loss,r_loss,accuracy_num / (iters+1))+str(learning_rate))

            ## Validation
            if (epoch + 1) % val_every==0:
                accuracy_num=0
                num=validation_data.shape[0]/batch_size
                epoch_loss=0
                for iters in range(0,validation_data.shape[0]/batch_size):                  
                    X,Y=get_normalized_data(iters,batch_size,validation_data,0)
                    c, a, pred = sess.run([loss,accuracy,prediction], feed_dict={x: X, y: Y, is_training: False,lr: 0})                   
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

train_neural_network(train_data)
