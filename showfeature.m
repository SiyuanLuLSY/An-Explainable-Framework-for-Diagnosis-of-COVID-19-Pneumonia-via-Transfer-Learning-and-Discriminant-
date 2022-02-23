%  load('nettrained.mat')
% Get the network weights for the second convolutional layer
w1 = mo.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5);

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
%montage(w1(:,:,5,:))
montage(w1)
% title('First convolutional layer weights')
