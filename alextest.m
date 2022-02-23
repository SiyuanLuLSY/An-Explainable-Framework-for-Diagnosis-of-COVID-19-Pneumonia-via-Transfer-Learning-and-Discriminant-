clear ;

  load net;
%    net=alexnet;
imds = imageDatastore('cmb', ...
   'IncludeSubfolders',true, ...
   'LabelSource','foldernames');

imdsTrain = imds;
% imdsTrain=imageDatastore('train', ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');
% imdsValidation=imageDatastore('test', ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = readimage(imdsTrain,idx(i));
%     imshow(I)
% end


% inputSize = net.Layers(1).InputSize
% layersTransfer = net.Layers(1:end-3);
inputSize = net(1).InputSize
layersTransfer = net(1:end-3);
numClasses = numel(categories(imdsTrain.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(256,'Name','fc8','WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    fullyConnectedLayer(numClasses,'Name','fc9','WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];

% lgraph = layerGraph(layers);
% figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
% plot(lgraph)

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

% augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

% [YPred,scores] = classify(netTransfer,augimdsValidation);

% idx = randperm(numel(imdsValidation.Files),4);

% YValidation = imdsValidation.Labels;
% accuracy = mean(YPred == YValidation)
% confMat = confusionmat(YValidation, YPred)
%     figure
%     cm = confusionchart(YValidation, YPred, ...
%     'Title','My Title', ...
%     'RowSummary','row-normalized', ...
%     'ColumnSummary','column-normalized');
layer = 'fc8';
tic
%提取训练图像fc7层数据
trainingFeatures = activations(netTransfer, augimdsTrain, layer, ...
'MiniBatchSize', 32, 'OutputAs', 'columns');
%提取测试图像fc7层数据
% testFeatures = activations(netTransfer,augimdsValidation,layer, ...
% 'MiniBatchSize', 32, 'OutputAs', 'columns');
toc
% testy = imdsValidation.Labels;
trainy = imdsTrain.Labels;

trainx=trainingFeatures;
% testx=testFeatures;
% save ('testx.mat','testx');
% save ('testy.mat','testy');
save ('trainx.mat','trainx');
save ('trainy.mat','trainy');

