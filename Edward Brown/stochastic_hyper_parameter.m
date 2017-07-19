function stochastic_hyper_parameter
    nTrain = 500;
    [X,y,X_test,y_test] = load_data(nTrain);
    X = horzcat(X, ones(size(y)));
    X_test = horzcat(X_test, ones(size(y_test)));
    
    % Parameters
    rng(41);
    learning_rates = 10.^[5,4,3,2,1,0,-1,-2,-3,-4];
    batch_sizes = [1,10,100,500];
    tol = 0.90;
    epoch_limit = 40;
    
   svm_params = struct('X', X, ...
    'y', y, ...
    'batch_size', 500, ...
    'epoch_limit', 500, ...
    'tol', 0.9, ...
    'learning_rate', 0.7, ...
    'X_test', X_test, ...
    'y_test', y_test, ...
    'by_batch', true);

    %% Gradient Descent for best lambda
    svm_params.epoch_limit = 50;
    svm_params.learning_rate = 1;
    
    svm_params.lambda = 10^-4;
    plot_by_batch(svm_params);
    
    %% Stochastic Gradient Descent
    lambda = 10^-5;
    
    for i = 1: length(batch_sizes)
        for j = 1:length(learning_rates)
            b  = primal_svm(X,y,batch_sizes(i), ...
                epoch_limit,lambda,tol,learning_rates(j),X_test,y_test,false);

            epochs_to_tol(i,j) = b.epochs_to_tol;
            final_costs(i,j) = b.costs_per_epoch(end);
            test_accuracies(i,j) = b.test_accuracy(end);
            train_accuracies(i,j) = b.training_accuracy(end);
            
        end
    end
      
    plot_heat(epochs_to_tol,'Epochs To Reach 0.90 Accuracy',batch_sizes,learning_rates);
    plot_heat(log(final_costs),'Log Costs',batch_sizes,learning_rates);
    plot_heat(test_accuracies,'Test Accuracy',batch_sizes,learning_rates);
    plot_heat(train_accuracies,'Train Accuracy',batch_sizes,learning_rates);
    
    svm_params.lmabda = 10^-6;
    svm_params.epoch_limit = 100;
    svm_params.learning_rate = 1; svm_params.batch_size = 100;
    plot_by_batch(svm_params)
end








