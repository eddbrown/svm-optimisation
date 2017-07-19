%% Hinge Loss Sub gradient derivative
function rtn = hld(w, X, y)
    dimensions = size(X,2);
    mult = y.*prediction(w,X);
    q = mult < 1;
    qs = repmat(q,1,dimensions);
    ys = repmat(y,1,dimensions);
    rtn = sum(-qs .* ys .* X,1);
end
