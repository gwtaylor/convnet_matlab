function [f, df] = fn_classify(VV,Dim,yy,targets)

% Given current setting of top-layer convnet parameters (VV) and input to
% top layer (yy)
% Compute cross-entropy error (f) using true targets (targets)
% Compute gradient of cost w.r.t. top-layer weights (df)
% Network dimensions are specified (Dim)
% Weights are inclusive of bias

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
