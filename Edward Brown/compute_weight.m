function w = compute_weight(alphas, y, X)
    [~, dimensions] = size(X);
    ys = repmat(y,1,dimensions);
    alphas_rep = repmat(alphas,1,dimensions);
    
    w = sum(ys.*alphas_rep.*X,1);
end

