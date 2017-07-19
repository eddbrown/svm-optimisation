function mnist_SVM_dual(redistribute)
    close all;
    tic
    % Load Preprocessed Data
    rng(42)
    [X,y,X_test,y_test] = load_data(1000);
    [no_test,~] = size(X_test);
    
    % Parameters
    rng(42);
    C = 0.02;
    [data_size, dimensions] = size(X);
    lambda = 2/(data_size * C);
    half_c = C/2;
    learning_rate = 0.0001;
    epochs = 60;
    tol = 10^-4;
    ys = repmat(y,1,dimensions);
    K = (ys.*X)*(ys .* X)';

    % Initialise weight/alphas vector
    alphas = half_c * ones(size(y));

    % Start SVM loop
    % --------------------------------------------------------------------     
    
    % Loop through data once
    % Perform iteration for number of epochs     
    for epoch_index = 1:epochs
        disp(epoch_index)
        for i = 1:data_size
            old_alpha = alphas(i);
            alpha_update = learning_rate * (1 - update_alpha(K(i,:),alphas,i));

            new_alpha = old_alpha + alpha_update;
            new_alpha = min(C, new_alpha);
            new_alpha = max(0, new_alpha);
            
            alphas(i) = new_alpha;
            if isnan(alpha_update)
                disp('ERROR')
            end
            
            if rem(i,1) == 0
                residual = y' * alphas;
                indices = find(alphas < C );
                alphas = alphas  - 1/sum(alphas) * (residual) * y .* alphas;
            end
        end
        
        w = compute_weight(alphas, y, X);
        bias = mean(y - (X * w'));
        train_acc(epoch_index) = mean(sign(prediction(w,X) + bias) == y);
        test_acc(epoch_index) = ...
            mean(sign(prediction(w,X_test)+ bias) == y_test);
    end

    % Output analysis
    figure;plot(test_acc)
    hold on;plot(train_acc)
    legend('Test', 'Train')
    toc
end





