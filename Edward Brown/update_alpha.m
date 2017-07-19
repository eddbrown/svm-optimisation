function rtn = update_alpha(kernel_entry,alphas,index)
    [data_size,~] = size(alphas);
    
    deltas = zeros(data_size,1);
    deltas(index) = 1;
    sum_terms = 0.5 * ((1 + deltas) .* alphas)' * kernel_entry';
    
    rtn =  sum_terms;
end