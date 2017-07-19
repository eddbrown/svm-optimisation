function rtn = primal_svm(X,y,batch_size, ...
    epoch_limit,lambda,stopping_threshold,learning_rate,X_test,y_test,...
    by_batch)

    [data_size, dimensions] = size(X);
    batches_per_epoch = data_size/batch_size;
    w = randn(1,dimensions);
    costs= [];
    test_acc = [];
    train_acc = [];
    costs_by_batch = [];
    test_acc_by_batch = [];
    train_acc_by_batch = [];
    
    epochs_to_reach_tol = 0;
    
    % Perform iteration for number of epochs     
    for epoch_index = 1:epoch_limit
%         disp(epoch_index)
        for batch_index = 1:batches_per_epoch
            % Prepare batch
            lower_bound = (batch_index - 1)* batch_size + 1;
            upper_bound = batch_index * batch_size;
            X_batch = X(lower_bound:upper_bound, :);
            y_batch = y(lower_bound:upper_bound);
            
            % Compute Derivative
            derivative = primal_derivative(w,X_batch,y_batch,lambda);
            
            % Update weight
            w = w - learning_rate * derivative/batch_size;
            
            if by_batch
                costs_by_batch = [costs_by_batch cost(w, X, y, lambda)];
                train_acc_by_batch = [train_acc_by_batch ... 
                    accuracy(X,y,w)];
                test_acc_by_batch = [test_acc_by_batch ... 
                    accuracy(X_test,y_test,w)];
            end
        end
        train_acc_epoch = accuracy(X,y,w);
        
        if train_acc_epoch > stopping_threshold && epochs_to_reach_tol == 0
            epochs_to_reach_tol = epoch_index;
        end
        
        train_acc = [train_acc train_acc_epoch];
        test_acc = [test_acc accuracy(X_test,y_test,w)];
        
        % Store total loss after epoch
        costs(epoch_index) = cost(w, X, y, lambda);
        
        if epoch_index == epoch_limit && epochs_to_reach_tol == 0
            epochs_to_reach_tol = epoch_limit;
        end
    end
    
    % Output analysis
%     acc = accuracy(prediction(w,X_test),y_test);
    rtn = struct('epochs', epoch_index, ...
        'costs_per_epoch', costs, ...
        'training_accuracy', train_acc, ...
        'test_accuracy', test_acc, ...
        'epochs_to_tol', epochs_to_reach_tol, ...
        'train_acc_by_batch', train_acc_by_batch, ...
        'test_acc_by_batch', test_acc_by_batch, ...
        'costs_by_batch', costs_by_batch);
end
