function targetout = convnet_probs(yy,w_class)

%multi-class logistic regression on yy
%returns probabilities
  
numlabels = size(w_class,2);
%should we change this here to deal with overflow/underflow probs?
targetout = exp(yy*w_class);
%targetout = targetout./repmat(sum(targetout,2),1,numlabels);
targetout = bsxfun(@rdivide,targetout,sum(targetout,2));