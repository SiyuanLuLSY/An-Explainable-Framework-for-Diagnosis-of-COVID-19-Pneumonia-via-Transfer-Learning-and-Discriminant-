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
    for k=1:n %������֤k=10��10����������Ϊ���Լ�
        test = (indices == k); %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
        train1 = ~test; %train��Ԫ�صı��Ϊ��testԪ�صı��
        XTrain=Input(:,:,:,train1); %�����ݼ��л��ֳ�train����������
        TTrain=Target(:,train1)'; %����������Ĳ���Ŀ��
        XTest=Input(:,:,:,test); %test������
        TTest=Target(:,test)';
        CMB(k)=sum(TTest=='true');
        nonCMB(k)=sum(TTest=='false');
    end
