%% Hinge loss function
function rtn = hinge(y,f)
    rtn = max(0,1-y.*f);
end

