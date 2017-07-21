import numpy as np
import random
import scipy.ndimage
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


def data_aug(img,aug_param,return_shift = 0):
    # random crop
    if aug_param['crop_size']:
        if aug_param['crop_size'] < img.shape:
            crop_len=[img.shape[i] -aug_param['crop_size'][i] for i in range(img.ndim)];
            r=[random.randint(aug_param['min_border'][i],crop_len[i]-aug_param['min_border'][i]) for i in range(img.ndim)]
            img=img[r[0]:r[0]+aug_param['crop_size'][0],r[1]:r[1]+aug_param['crop_size'][1],r[2]:r[2]+aug_param['crop_size'][2]]
        else:
            print 'crop_size is larger than minimum image shape, that is %d >= %d' %(aug_param.crop_size,min(img.shape))
    
    # random flip
    if aug_param['flip']:
        if random.randint(0,1):
            img=np.fliplr(img)
        if random.randint(0,1):
            img=np.flipud(img)
            
    # random rot
    if aug_param['rot']:
        k=random.randint(0,4)
        if k:
            img=[np.rot90(img[i],k) for i in range(img.shape[0])]     
    img=np.array(img)
    
    # blur or sharpen
    if aug_param['blur_sigma']:
        img=img.astype(float)
        if aug_param['sharpen'] and random.randint(0,1):
            blured = scipy.ndimage.gaussian_filter(img, sigma=random.uniform(0,aug_param['blur_sigma']))
            img=img+(img-blured)
        else:
            img = scipy.ndimage.gaussian_filter(img, sigma=random.uniform(0,aug_param['blur_sigma']))

    # randome noise
    if aug_param['noise_max_amplitude']:
        amplitude=random.randint(0,aug_param['noise_max_amplitude'])
        img=img.astype(float)
        img =img+ (np.random.random(img.shape)*amplitude-amplitude/2)
    if return_shift:
        return img,r
    else:
        return img
    
def data_aug_2d(img,aug_param,return_shift = 0):
    # random crop
    if aug_param['zoom'] > 1:
        zoom_x = random.random() * (aug_param['zoom'] - 1) + 1
        zoom_y = random.random() * (aug_param['zoom'] - 1) + 1
        img = scipy.ndimage.interpolation.zoom(img,[zoom_x,zoom_y,1])
    if aug_param['crop_size']:
        if aug_param['crop_size'] < img.shape:
            crop_len=[img.shape[i] -aug_param['crop_size'][i] for i in range(img.ndim)];
            if aug_param['random_crop'] == True:
                try:
                    r=[random.randint(aug_param['min_border'][i],crop_len[i]-aug_param['min_border'][i]) for i in range(img.ndim)]
                except:
                    print crop_len,img.shape[i],aug_param['crop_size']
            else:
                r=[crop_len[i]/2 for i in range(img.ndim)]
            img=img[r[0]:r[0]+aug_param['crop_size'][0],r[1]:r[1]+aug_param['crop_size'][1]]
        else:
            print 'crop_size is larger than minimum image shape, that is %d >= %d' %(aug_param.crop_size,min(img.shape))
    
    # random flip
    if aug_param['flip']:
        if random.randint(0,1):
            img=np.fliplr(img)
        if random.randint(0,1):
            img=np.flipud(img)
            
    # random rot
    if aug_param['rot']:
        k=random.randint(0,4)
        if k:
            img=np.rot90(img,k)
    img=np.array(img)
    
    # blur or sharpen
    if aug_param['blur_sigma']:
        img=img.astype(float)
        if aug_param['sharpen'] and random.randint(0,1):
            blured = scipy.ndimage.gaussian_filter(img, sigma=random.uniform(0,aug_param['blur_sigma']))
            img=img+(img-blured)
        else:
            img = scipy.ndimage.gaussian_filter(img, sigma=random.uniform(0,aug_param['blur_sigma']))

    # randome noise
    if aug_param['noise_max_amplitude']:
        amplitude=random.randint(0,aug_param['noise_max_amplitude'])
        img=img.astype(float)
        img =img+ (np.random.random(img.shape)*amplitude-amplitude/2)
    if return_shift:
        return img,r
    else:
        return img
    
if __name__ == "__main__":
	data=np.load('sample2.npy')
	img=data[12,0]
	train_aug_params = {
    'crop_size': 48,
    'sharpen': True,
    'blur_sigma': 0.5,
    'noise_max_amplitude': 40,
    'flip': True,
    'rot': True,
    }
	t1=time.time()
	img=data_aug(img,train_aug_params)
	print time.time()-t1
	a=img[32]
	plt.imshow(img[32])
                            
                            