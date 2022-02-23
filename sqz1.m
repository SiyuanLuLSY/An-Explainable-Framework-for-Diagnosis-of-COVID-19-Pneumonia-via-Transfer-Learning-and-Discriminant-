function [trainingFeatures,trainy,testFeatures,testy]=sqz1(net,layer)%for training set
%clear;
%unzip('MerchData.zip');
imds = imageDatastore('W3TEST', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsTrain = imds;

numTrainImages = numel(imdsTrain.Labels);
% idx = randperm(numTrainImages,16);

net.Layers
% lgraph = layerGraph(net);
% figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
% plot(lgraph)
 inputSize = net.Layers(1).InputSize
% layersTransfer = net.Layers(1:end-1);
% numClasses = numel(categories(imdsTrain.Labels))
% 
% lgraph = removeLayers(lgraph, {'ClassificationLayer_fc1000','fc1000_softmax','fc1000'});
% 
% numClasses = numel(categories(imdsTrain.Labels));
% nlayers = [
%     fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','classoutput')];
% lgraph = addLayers(lgraph,nlayers);
% lgraph = connectLayers(lgraph,'avg_pool','fc');

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

% augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
%     layer = 'fc8';
    %提取训练图像fc7层数据
    trainingFeatures = activations(net, augimdsTrain, layer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
    %提取测试图像fc7层数据
%     testFeatures = activations(net,augimdsValidation,layer, ...
%     'MiniBatchSize', 32, 'OutputAs', 'columns');

trainy = imdsTrain.Labels;

end