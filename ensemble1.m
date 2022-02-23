n=10;
% sen=zeros(1,n);
% spe=zeros(1,n);
acc=zeros(1,n);
N=1000;
tic
for i=1 : n
    [ind,TTest]=entest(N,trainx,trainy,testx,testy);
    [m,order] = confusionmat(TTest,single(ind));
    acc(i)=(m(1,1)+m(2,2))/sum(sum(m))
    figure
    cm = confusionchart(TTest,single(ind), ...
    'Title','My Title', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
end
toc
 mean_acc=mean(acc)
