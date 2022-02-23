function [net,confMat]=train_net(XTrain,TTrain,XTest,TTest)


  layers = [imageInputLayer([41 41 1],'Normalization','zerocenter')
          convolution2dLayer(3,32,'Padding',2)
          batchNormalizationLayer('Name','BN1')
          reluLayer()
          maxPooling2dLayer(3,'Stride',1)
          convolution2dLayer(3,64,'Padding',2)
          batchNormalizationLayer('Name','BN2')
          reluLayer()
          convolution2dLayer(5,64,'Padding',2)
          batchNormalizationLayer('Name','BN3')
          reluLayer()
          maxPooling2dLayer(7,'Stride',1)
          fullyConnectedLayer(64,'Name','fc1')
          reluLayer('Name','relu')
          dropoutLayer(0.5)
          fullyConnectedLayer(2,'Name','fc2')
          softmaxLayer('Name','softmax')
          classificationLayer()];
options = trainingOptions('rmsprop', ...
    'MiniBatchSize',200, ...
    'MaxEpochs',2, ...
    'InitialLearnRate',1e-4, ...
    'L2Regularization',0.001,...
    'ExecutionEnvironment','gpu',...
    'Verbose',false, ...
    'ValidationData',{XTest,TTest},...
    'ValidationFrequency',40,...
    'Plots','training-progress');        
net = trainNetwork(XTrain,TTrain,layers,options);
     
YTest = classify(net,XTest);

confMat = confusionmat(TTest, YTest)
accuracy = sum(YTest == TTest)/numel(TTest)
%     figure
%     cm = confusionchart(TTest,YTest, ...
%     'Title','My Title net', ...
%     'RowSummary','row-normalized', ...
%     'ColumnSummary','column-normalized');

% YTrain = classify(net,XTrain);
% confMat = confusionmat(TTrain, YTrain)
% accuracy = sum(YTrain == TTrain)/numel(TTrain)

% featureLayer='fc1';
% tic 
% trainx = activations(net, XTrain, featureLayer, ...
%     'MiniBatchSize', 32, 'OutputAs', 'columns');
% toc
% trainy=TTrain;
% testx=activations(net, XTest, featureLayer, ...
%     'MiniBatchSize', 32, 'OutputAs', 'columns');
% testy=TTest;

