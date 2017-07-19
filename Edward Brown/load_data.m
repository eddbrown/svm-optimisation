function [X,y,X_test,y_test] = load_data(N_train)
    train = load('X.mat');
    train = train.X;
    X = train(:,1:784);
    y = train(:,785);
    indices = randperm(size(X,1))';
    indices = indices(1:N_train,:);
    X = X(indices,:);
    y = y(indices);
    
    test = load('X_test.mat');
    test = test.X_test;
    X_test = test(:,1:784);
    y_test = test(:,785);
end
