function rtn = primal_derivative(w, X, y, lambda)
    data_size = size(X,1);  
    rtn = data_size * lambda * w + hld(w,X,y);
end

