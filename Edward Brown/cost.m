%% Cost Function
function rtn = cost(w, X, y, lambda)
    [data_size, ~] = size(X);
    f = prediction(w, X);
    hinge_sum = sum(hinge(y,f));
    
    rtn = (lambda/2) * norm(w)^2 + (1/data_size) * hinge_sum;
end