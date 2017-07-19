function K = kernel_matrix(X,X2,params)
sigma = params.sigma;
ker = params.kernel;
b = params.b;
d = params.d;
switch ker
    case 'lin'
        K = X' * X2;
      

    case 'poly'
        if exist('X2','var')
            K = (X' * X2 + b).^d;
        else
            K = (X' * X + b).^d;
        end

    case 'rbf'
        n1sq = sum(X.^2,1);
        n1 = size(X,2);

        if isempty(X2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end;
        K = exp(-D/(2*sigma^2));

    case 'sam'
        if exist('X2','var');
            D = X'*X2;
        else
            D = X'*X;
        end
        K = exp(-acos(D).^2/(2*sigma^2));

    otherwise
        error(['Unsupported kernel ' ker])
end
