function [f, df] = CG_SMALLNORB_CLASSIFY_CINIT(VV,Dim,yy,targets)

%do backprop only on the classification weights, w_class
%note that these weights include a bias

l1 = Dim(1);
l2 = Dim(2);

N = size(yy,1);
% Do deconversion.
w_class = reshape(VV,l1+1,l2);
yy = [yy ones(N,1)];  

targetout = convnet_probs(yy,w_class);

f = -sum(sum( targets(:,1:end).*log(targetout))) ;
delta3 = (targetout-targets(:,1:end));
dw_class =  yy'*delta3; 

df = dw_class(:);
