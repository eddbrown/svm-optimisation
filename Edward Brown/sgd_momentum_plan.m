function info = sgd_momentum_plan(X,y,X_test,y_test,w0,opts)
    batch_size = opts.batch_size;
    lambda = opts.lambda;
    learning_rate = opts.learning_rate;
    max_iter = opts.max_iter;
    gamma_init = opts.gamma_init;
    gamma_final = opts.gamma_final;
    method = opts.method;
    [data_size, ~] = size(X);
    batches_per_epoch = data_size/batch_size;          
    w = w0;
    v = zeros(size(w));
    iter = 0;
    b = log(1/(gamma_final - gamma_init));
    
    test_acc = accuracy(X_test, y_test, w);
    test_accs = test_acc;
    cost_before = cost(w, X, y, lambda);
    costs = cost_before;
    
    % Perform iteration for number of epochs     
    while iter <= max_iter
        for batch_index = 1:batches_per_epoch 
            % Prepare batch
            lower_bound = (batch_index - 1)* batch_size + 1;
            upper_bound = batch_index * batch_size;
            X_batch = X(lower_bound:upper_bound, :);
            y_batch = y(lower_bound:upper_bound);
            
            % Compute Derivative
            derivative = primal_derivative(w,X_batch,y_batch,lambda);
            
            % Update weight
            switch method
                case 'normal'
                    gamma = 0;
                case 'momentum'
                    gamma = gamma_init;
                case 'plan'
                    gamma = gamma_final - exp(-(0.001*(iter) + b));
            end
            v = gamma * v + learning_rate * derivative/batch_size;
            w = w - v;
        end
 
        % Store total loss after epoch
        cost_after = cost(w, X, y, lambda);
        costs = [costs cost_after];
        
        norm_error = norm(cost_after - cost_before)/norm(cost_before);
        cost_before = cost_after;
        
        test_acc = accuracy(X_test, y_test, w);
        test_accs = [test_accs test_acc];

        
        iter = iter + 1;
    end
    
    info.costs = costs;
    info.nIter = iter;
    info.test_accs = test_accs;
end
