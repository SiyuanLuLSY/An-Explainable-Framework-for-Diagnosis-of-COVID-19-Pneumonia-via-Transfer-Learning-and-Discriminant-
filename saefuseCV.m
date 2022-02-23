load('fused.mat');

Target=categorical(Target);
a=Input(1:600,:);
res=Input(601:1200,:);
m=Input(1201:1800,:);
Input=[a; m];

N_length=length(Target);
N=20;
n=5;
nconfm=zeros(2,2,n);
m=zeros(2,2,n);
tic
    indices=crossvalind('Kfold',N_length,n);
    for k=1:n %������֤k=10��10����������Ϊ���Լ�
        test = (indices == k); %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
        train1 = ~test; %train��Ԫ�صı��Ϊ��testԪ�صı��
        XTrain=Input(:,train1); %�����ݼ��л��ֳ�train����������
        TTrain=Target(train1,:); %����������Ĳ���Ŀ��
        XTest=Input(:,test); %test������
        TTest=Target(test,:);
        [Ax,Ay,Xs,Ys]=dcaFuse(XTrain(1:600,:),XTrain(601:1200,:),TTrain');
       XTrain=[Xs;Ys];
%         XTrain=Xs+Ys;
        x=Ax*XTest(1:600,:);y=Ay*XTest(601:1200,:);
       XTest=[x;y];
%         XTest=x+y;
       [cfmelm(:,:,k),cfmsnn(:,:,k),cfmrvfl(:,:,k),cfmen(:,:,k),...
           ta,TTest]=entest(N,XTrain,TTrain,XTest,TTest);
       [sensitivity_net(k),specificity_net(k),accuracy_net(k),...
           precision_net(k),F1_net(k)]=getindexes(cfmen(:,:,k));
    end
ttime=toc
m_acc=mean(accuracy_net)
m_sen=mean(sensitivity_net)
m_spe=mean(specificity_net)
m_pre=mean(precision_net)
m_F1=mean(F1_net)