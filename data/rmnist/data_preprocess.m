%clear;
load('data/mnist_rotation_train.amat', '-ascii')
load('data/mnist_rotation_test.amat', '-ascii')
%% 
x=mnist_rotation_train(:,1:784);
y=mnist_rotation_train(:,785);
num=size(x,1);
im_mean=mean(x);
im_std=std(x);
x=(x-repmat(im_mean,num,1))./repmat(im_std,num,1);
x=reshape(x,num,1,28,28);

x_test=mnist_rotation_test(:,1:784);
y_test=mnist_rotation_test(:,785);
num=size(x_test,1);
im_mean=mean(x_test);
im_std=std(x_test);
x_test=(x_test-repmat(im_mean,num,1))./repmat(im_std,num,1);
x_test=reshape(x_test,num,1,28,28);

save('data.mat','x','y','x_test','y_test')