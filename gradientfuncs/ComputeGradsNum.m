function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(1,lenght(W));
grad_b = cell(1,length(b));

d = size(X, 1);
for k=1:length(W)
    no = size(W{k}, 1);
    
    
    grad_W{k} = zeros(size(W{k}));
    grad_b{k} = zeros(no, 1);
    
    c = ComputeCost(X, Y, W{k}, b{k}, lambda);
    
    for i=1:length(b)
        b_try = b{k};
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W{k}, b_try, lambda);
        grad_b{k}(i) = (c2-c) / h;
    end
    
    for i=1:numel(W)
        
        W_try = W{k};
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b{k}, lambda);
        
        grad_W{k}(i) = (c2-c) / h;
    end
end