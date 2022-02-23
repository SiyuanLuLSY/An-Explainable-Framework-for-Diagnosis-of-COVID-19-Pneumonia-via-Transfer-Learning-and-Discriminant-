function [sensitivity,specificity,accuracy,precision,F1]=getindexes(cfm)
% false true
TP=cfm(2,2); FP=cfm(2,1);TN=cfm(1,1);FN=cfm(1,2);
sensitivity=TP/(TP+FN);
specificity=TN/(TN+FP);
accuracy=(TP+TN)/(TP+TN+FP+FN);
precision=TP/(TP+FP);
F1=2*(precision*sensitivity)/(precision+sensitivity);