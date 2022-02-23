clear;
n=41; t=13031;
nut=10000;
load('Dataset_41_13031.mat');
new=zeros(n,n,1,t);
new(:,:,1,:)=Input;
Input=new;

XTrain = Input(:,:,:,1:nut);
XTest = Input(:,:,:,nut+1:t);
TTrain = categorical(Target(:,1:nut));
TTrain = TTrain';
YTest = categorical(Target(:,nut+1:t));
TTest =YTest'; 

% XTrain = Input(:,:,:,1:t);
% TTrain = categorical(Target(:,1:t));
% TTrain = TTrain';


  layers = [imageInputLayer([n n 1],'Normalization','zerocenter')
          convolution2dLayer(3,64,'Padding',2)
          batchNormalizationLayer('Name','BN1')
          reluLayer()
          maxPooling2dLayer(2,'Stride',1)
          convolution2dLayer(3,128,'Padding',2)
          batchNormalizationLayer('Name','BN2')
          reluLayer()
          maxPooling2dLayer(2,'Stride',1)
          fullyConnectedLayer(128,'Name','fc1')
          reluLayer('Name','relu')
          dropoutLayer(0.5)
          fullyConnectedLayer(2,'Name','fc2')
          softmaxLayer('Name','softmax')
          classificationLayer()];
options = trainingOptions('rmsprop', ...
    'MiniBatchSize',200, ...
    'MaxEpochs',3, ...
    'InitialLearnRate',1e-4, ...
    'L2Regularization',0.001,...
    'ExecutionEnvironment','gpu',...
    'Verbose',false, ...
    'ValidationData',{XTest,TTest},...
    'ValidationFrequency',20,...
    'Plots','training-progress');        
net = trainNetwork(XTrain,TTrain,layers,options);
      
YTest = classify(net,XTest);
confMat = confusionmat(TTest, YTest)
accuracy = sum(YTest == TTest)/numel(TTest)

% YTrain = classify(net,XTrain);
% confMat = confusionmat(TTrain, YTrain)
% accuracy = sum(YTrain == TTrain)/numel(TTrain)

featureLayer='fc1';
trainx = activations(net, Input, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
trainy=TTrain;
testy=TTest;
