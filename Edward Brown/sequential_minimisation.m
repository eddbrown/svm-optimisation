function sequential_minimisation
% SVM using Sequential Minimal Optimization (SMO)
% The algorithm was written with help from this resource:
% https://uk.mathworks.com/matlabcentral/
% fileexchange/63100-smo--sequential-minimal-optimization-

rng(42)
[x_train,y_train,x_test,y_test] = load_data(1000);
x_train = horzcat(x_train, ones(size(y_train)));
x_test = horzcat(x_test, ones(size(y_test)));
N = length(y_train);
C = 0.02;
tol = 0.0003;
alpha = zeros(N,1);
bias = 0;
x = x_train;
y = y_train;
nought_count = 0;
changed_alpha_plot = [];

%  SMO Algorithm
while (1)
    changed_alphas = 0;
    N = size(y,1);
    K = x * x';
    for i=1:N
        Ei=sum(alpha.*y.*K(:,i))-y(i);
        if ((Ei*y(i)<-tol) && alpha(i)<C)|| ...
                (Ei*y(i) > tol && (alpha(i) > 0))
            for j=[1:i-1,i+1:N]
                Ej=sum(alpha.*y.*K(:,j))-y(j);
                  alpha_iold=alpha(i);
                  alpha_jold=alpha(j);

                  if y(i)~=y(j)
                      L=max(0,alpha(j)-alpha(i));
                      H=min(C,C+alpha(j)-alpha(i));
                  else 
                      L=max(0,alpha(i)+alpha(j)-C);
                      H=min(C,alpha(i)+alpha(j));
                  end

                  if (L==H)
                      continue
                  end

                  eta = 2*x(j,:)*x(i,:)'-x(i,:)*x(i,:)'-x(j,:)*x(j,:)';

                  if eta>=0
                      continue
                  end

                  alpha(j)=alpha(j)-( y(j)*(Ei-Ej) )/eta;
                  if alpha(j) > H
                      alpha(j) = H;
                  end
                  if alpha(j) < L
                      alpha(j) = L;
                  end

                  if norm(alpha(j)-alpha_jold,2) < tol
                      continue
                  end

                  alpha(i)=alpha(i)+y(i)*y(j)*(alpha_jold-alpha(j));
                  b1 = bias - Ei - y(i)*(alpha(i)-alpha_iold)*x(i,:)*...
                      x(i,:)'-y(j)*(alpha(j)-alpha_jold)*x(i,:)*x(j,:)';
                  b2 = bias - Ej - y(i)*(alpha(i)-alpha_iold)*x(i,:)*...
                      x(j,:)'-y(j)*(alpha(j)-alpha_jold)*x(j,:)*x(j,:)';

                  changed_alphas = changed_alphas+1;
            end
        end
    end
    changed_alpha_plot = [changed_alpha_plot changed_alphas];
    if changed_alphas == 0
        nought_count = nought_count + 1;
    else
        nought_count = 0;
    end

    if nought_count == 3
        break
    end

    x=x((find(alpha~=0)),:);
    y=y((find(alpha~=0)),:);
    alpha=alpha((find(alpha~=0)),:);
end

% Test Accuracy
w = compute_weight(alpha, y, x);
bias = mean(y - (x * w'));
train_accuracy = mean(sign(prediction(w,x_train) + bias) == y_train)
test_accuracy = mean(sign(prediction(w,x_test) + bias) == y_test)
figure;
plot(changed_alpha_plot)
ylabel('Alphas Changed')
xlabel('Passes')
title('Alpha values changes')
end