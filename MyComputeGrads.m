%mygrad
function [grad_W,grad_b]=MyComputeGrads(X,Y,W,b,lambda)
N = size(X,2);
P = EvaluateClassifier(X,W,b);
dldb = zeros(size(b));
dldW = zeros(size(W));
for i=1:N
    g = (-Y(:,i)'/(Y(:,i)'*P(:,i)))*(diag(P(:,i))-(P(:,i)*P(:,i)'));
    dldb = (dldb + g');
    dldW = (dldW + g'*X(:,i)');
end

grad_W = dldW/N + 2*lambda*W;
grad_b = dldb/N;
end