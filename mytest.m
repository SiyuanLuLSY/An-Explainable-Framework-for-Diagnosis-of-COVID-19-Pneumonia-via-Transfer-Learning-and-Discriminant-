n=10;
sen=zeros(1,n);
spe=zeros(1,n);
acc=zeros(1,n);

for i=1 : n
    [ind,TTest,acc(i),testtime]=CBAtest(trainx,trainy,testx,testy,1000);
%     [m,order] = confusionmat(TTest,single(ind));
%     figure
%     cm = confusionchart(TTest,single(ind), ...
%     'Title','My Title', ...
%     'RowSummary','row-normalized', ...
%     'ColumnSummary','column-normalized');
end
% testtime=testtime/88;
% tic
% %��ȡѵ��ͼ��fc7������
% trainingFeatures = activations(netTransfer, augimdsTrain, layer, ...
% 'MiniBatchSize', 32, 'OutputAs', 'columns');
% %��ȡ����ͼ��fc7������
% testFeatures = activations(netTransfer,augimdsValidation,layer, ...
% 'MiniBatchSize', 32, 'OutputAs', 'columns');
% time=toc;
% TT=time/296+testtime
mean_acc=mean(acc)
