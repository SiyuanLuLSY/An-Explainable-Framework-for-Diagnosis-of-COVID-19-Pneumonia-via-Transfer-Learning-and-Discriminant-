function lgraph=getres50(nout)


net = resnet50;
% net.Layers
lgraph = layerGraph(net);
% figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
% plot(lgraph)
%  inputSize = net.Layers(1).InputSize
% layersTransfer = net.Layers(1:end-1);
% numClasses = numel(categories(imdsTrain.Labels))
% 
lgraph = removeLayers(lgraph, {'ClassificationLayer_fc1000','fc1000_softmax','fc1000'});
% 

nlayers = [
    fullyConnectedLayer(256,'Name','fc256','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    reluLayer('Name','relu_out')
    fullyConnectedLayer(nout,'Name','fc2222','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax_out')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,nlayers);
lgraph = connectLayers(lgraph,'avg_pool','fc256');