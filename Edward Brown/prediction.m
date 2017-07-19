%% Prediction function
function rtn = prediction(weight,data)
    weights_big = repmat(weight,size(data,1),1);
    rtn = sum((data .* weights_big),2);
end
