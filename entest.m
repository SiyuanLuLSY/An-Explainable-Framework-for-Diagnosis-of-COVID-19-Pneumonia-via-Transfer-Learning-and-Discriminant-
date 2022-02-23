function [cfmelm,cfmsnn,cfmrvfl,cfmen,ta,TTest]=entest(N,trainx,trainy,testx,testy)
% clear;
% 
% load('testx.mat');
% load('testy.mat');
% load('trainx.mat');
% load('trainy.mat');
%N=400;
XTrain = trainx;
XTest = testx;

TTrain = single(trainy);
% TTrain = TTrain;
TTest = single(testy);
% TTest =YTest;


M1=[TTrain,XTrain'];
M2=[TTest,XTest'];
tic

[TY1,FP1,FN1,TrainingTime1, TestingTime1, TrainingAccuracy1, TestingAccuracy1] = elm(M1,...
    M2, 1, N, 'sig');

[TY2,FP2,FN2,TrainingTime2, TestingTime2, TrainingAccuracy2, TestingAccuracy2] = snn(M1,...
    M2, 1, N, 'sig');

option.ActivationFunction='sigmoid';option.N=N;
[train_accuracy3,TestingAccuracy3,TY3]=RVFL_train_val(XTrain',...
    TTrain,XTest',TTest,option);
% res=sum(abs(TY-TTest));

TY1=TY1';
[b1,ind1]=max(TY1,[],2);
cfmelm = confusionmat(TTest, single(ind1));
TY2=TY2';
[b2,ind2]=max(TY2,[],2);
cfmsnn = confusionmat(TTest, single(ind2));
ind3=TY3;
cfmrvfl = confusionmat(TTest, single(ind3));
ind=[ind1 ind2 ind3];
ta=0;
for i=1:size(ind,1)
table = tabulate(ind(i,:));
[maxCount,idx] = max(table(:,2));
y=table(idx);
if maxCount==1 y=1;end
ta=[ta;y];
end
ta=ta(2:end);
ta=ta';
toc
cfmen = confusionmat(TTest, single(ta));
