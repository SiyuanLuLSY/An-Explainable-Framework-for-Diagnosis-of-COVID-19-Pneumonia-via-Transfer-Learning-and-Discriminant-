clear;
net=alexnet;
layer='fc8';
% Logits_softmax Logits global_average_pooling2d_1 out_relu
% fc1000_softmax fc1000 avg_pool
% prob pool10 relu_conv10 conv10
% fc1000_softmax fc1000 avg_pool
% [trainx,trainy]=RES1(net,layer);
% [testx,testy]=RES2(net,layer);
[trainx,trainy,testx,testy]=sqz1(net,layer);