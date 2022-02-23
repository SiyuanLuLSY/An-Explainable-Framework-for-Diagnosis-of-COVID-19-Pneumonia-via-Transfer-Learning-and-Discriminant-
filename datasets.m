load('Dataset_41_13031.mat');
load('inds.mat');
% new=zeros(n,n,1,t);
% new(:,:,1,:)=Input;
% Input=new;
Target=categorical(Target);

N_length=length(Target);
N=800;
n=5;

tic
%     indices=crossvalind('Kfold',N_length,n);
    for k=1:n %交叉验证k=10，10个包轮流作为测试集
        test = (indices == k); %获得test集元素在数据集中对应的单元编号
        train1 = ~test; %train集元素的编号为非test元素的编号
        XTrain=Input(:,:,:,train1); %从数据集中划分出train样本的数据
        TTrain=Target(:,train1)'; %获得样本集的测试目标
        XTest=Input(:,:,:,test); %test样本集
        TTest=Target(:,test)';
        CMB(k)=sum(TTest=='true');
        nonCMB(k)=sum(TTest=='false');
    end
