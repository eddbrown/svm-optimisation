function acc = accuracy(X,y,w)
    acc = mean(sign(prediction(w,X)) == y);
end

