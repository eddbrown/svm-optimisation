function plot_by_batch(svm_params)
    X = svm_params.X;
    y = svm_params.y;
    batch_size = svm_params.batch_size;
    epoch_limit = svm_params.epoch_limit;
    lambda = svm_params.lambda;
    tol = svm_params.tol;
    learning_rate = svm_params.learning_rate;
    X_test = svm_params.X_test;
    y_test = svm_params.y_test;
    by_batch = svm_params.by_batch;
    
    svm_data = primal_svm(X,y,batch_size,epoch_limit,lambda,...
        tol,learning_rate,X_test,y_test,by_batch);
    
    costs_by_batch = svm_data.costs_by_batch;
    test_accuracy_by_batch = svm_data.test_acc_by_batch;
    train_accuracy_by_batch = svm_data.train_acc_by_batch;
    
    figure; plot(log(costs_by_batch)); title('Log Costs by Batch');
    xlabel('Batch');
    ylabel('Log Cost');
    dim = [.2 .5 .3 .3];
    str = sprintf('Batch Size: %d \n Learning Rate: %f \n Lambda: %f', batch_size, learning_rate, lambda);
    annotation('textbox',dim,'String',str,'FitBoxToText','on');
    
    figure; plot(test_accuracy_by_batch);
    str = sprintf('Batch Size: %d \n Learning Rate: %f \n Lambda: %f \n', batch_size, learning_rate, lambda);
    annotation('textbox',[.5 .5 .1 .1],'String',str,'FitBoxToText','on');
    title('Train And Test Accuracy by Batch'); hold on;
    plot(train_accuracy_by_batch);
    legend('Test Accuracy', 'Train Accuracy')
    xlabel('Batch');
    ylabel('Accuracy');
end
