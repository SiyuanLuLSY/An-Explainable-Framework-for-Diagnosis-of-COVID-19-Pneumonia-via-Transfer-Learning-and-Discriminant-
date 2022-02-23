function layers=getalex(num)
  net=alexnet;
layersTransfer = net.Layers(1:end-3);

layers = [
    layersTransfer
    fullyConnectedLayer(256,'Name','fc256','WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    reluLayer()
    fullyConnectedLayer(num,'Name','fc9','WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];