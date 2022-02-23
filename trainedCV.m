imds = imageDatastore('.\COVID_3','IncludeSubfolders',true,'LabelSource','foldernames');
%imds = imageDatastore('.\W3TEST','IncludeSubfolders',true,'LabelSource','foldernames');
[imd1 imd2 imd3 imd4 imd5] = splitEachLabel(imds,0.2,0.2,0.2,0.2,0.2,'randomize');
partStores{1} = imd1.Files ;
partStores{2} = imd2.Files ;
partStores{3} = imd3.Files ;
partStores{4} = imd4.Files ;
partStores{5} = imd5.Files ;

N=50;
n=5;
nconfm=zeros(2,2,n);
m=zeros(2,2,n);
idx = crossvalind('Kfold', n, n);
numClasses = numel(categories(imds.Labels));


layer='fc256';
% net_a=getalex(numClasses);  
net_mo=getres50(numClasses);
net_res=getres(numClasses);

  inputSize = net_mo.Layers(1).InputSize;  
  inputSize2 = net_res.Layers(1).InputSize;  
  for k = 1:n
        k
      test_idx = (idx == k);
      train_idx = ~test_idx;
      imdsTest = imageDatastore(partStores{test_idx}, 'IncludeSubfolders', true,'LabelSource', 'foldernames');
      imdsTrain = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true,'LabelSource', 'foldernames');
      augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'ColorPreprocessing','gray2rgb');
      augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsTest,'ColorPreprocessing','gray2rgb');

option = trainingOptions('rmsprop', ...
    'MiniBatchSize',50, ...
    'MaxEpochs',1, ...    
    'ExecutionEnvironment','gpu',...
    'InitialLearnRate',1e-4, ...     
    'ValidationData',augimdsValidation,...    
    'ValidationFrequency',10,...
    'Plots','training-progress',...
    'Verbose',false);
tic
      mo = trainNetwork(augimdsTrain,net_mo,option);
      [YPred,scores] = classify(mo,augimdsValidation);  
      cfmmo(:,:,k) = confusionmat(imdsTest.Labels, YPred)
%       YTest = classify(mo,augimdsValidation);
%       figure
%       cm = confusionchart(imdsTest.Labels,YTest, ...
%         'Title','My Title net', ...
%         'RowSummary','row-normalized', ...
%         'ColumnSummary','column-normalized');      
    
    res = trainNetwork(augimdsTrain,net_res,option);
      [YPred,scores] = classify(res,augimdsValidation);  
      cfmres(:,:,k) = confusionmat(imdsTest.Labels, YPred)
time1=toc;
tic
      f_a_train = activations(mo, augimdsTrain, layer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
     f_a_test = activations(mo,augimdsValidation,layer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
     f_res_train = activations(res, augimdsTrain, layer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
     f_res_test = activations(res,augimdsValidation,layer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
    TTest = imdsTest.Labels;
    TTrain = imdsTrain.Labels;
    TTest=categorical(TTest);
    TTrain=categorical(TTrain);
            [Ax,Ay,Xs,Ys]=dcaFuse( f_a_train, f_res_train,TTrain');
%        XTrain=[Xs;Ys];
        XTrain=Xs+Ys;
        x=Ax*f_a_test;y=Ay*f_res_test;
%        XTest=[x;y];
        XTest=x+y;
time2=toc;
tic
[cfmelm(:,:,k),cfmsnn(:,:,k),cfmrvfl(:,:,k),cfmen(:,:,k),...
           ta,TTest]=entest(N,XTrain,TTrain,XTest,TTest);
       cfmelm(:,:,k),cfmsnn(:,:,k),cfmrvfl(:,:,k),cfmen(:,:,k)
%        [sensitivity_net(k),specificity_net(k),accuracy_net(k),...
%            precision_net(k),F1_net(k)]=getindexes(cfmen(:,:,k));
  time3=toc;
  end
% m_acc=mean(accuracy_net)
% m_sen=mean(sensitivity_net)
% m_spe=mean(specificity_net)
% m_pre=mean(precision_net)
% m_F1=mean(F1_net)
  