function [xMin, fMin, nIter, info] = nonlinearConjugateGradient(F, ls, type, alpha0, x0, tol, maxIter)
% NONLINEARCONJUGATEGRADIENT Wrapper function executing conjugate gradient with Fletcher Reeves algorithm
% [xMin, fMin, nIter, info] = nonlinearConjugateGradient(F, ls, 'type', alpha0, x0, tol, maxIter) 
%
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - f2f: Hessian handler
% ls: handle to linear search function
% type: beta update type {'FR'}
% alpha0: initial step length 
% rho: in (0,1) backtraking step length reduction factor
% c1: constant in sufficient decrease condition f(x_k + alpha_k*p_k) > f_k + c1*alpha_k*(df_k')*p_k)
%     Typically chosen small, (default 1e-4). 
% x0: initial iterate
% tol: stopping condition on relative error norm tolerance 
%      norm(x_Prev - x_k)/norm(x_k) < tol;
% maxIter: maximum number of iterations
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% info: structure with information about the iteration 
%   - xs: iterate history 
%   - alphas: step lengths history 
%
% Copyright (C) 2017  Kiko Rullan, Marta M. Betcke 

% Initialization
nIter = 0;
normError = 1;
x_k = x0;
df_k = F.df(x_k);
p_k = -df_k;
info.cost = [];
info.alphas = alpha0;

% Loop until convergence or maximum number of iterations
while (normError >= tol && nIter <= maxIter)
    % Call line search given by handle ls for computing step length
    alpha_k = ls(x_k, p_k, alpha0);
    
    % Update x_k and df_k
    x_k_1 = x_k;
    x_k = x_k + alpha_k*p_k;
    df_k_1 = df_k;
    df_k = F.df(x_k);
    % Compute descent direction
    switch upper(type)
      case 'FR'
          beta_k = (df_k*df_k')/(df_k_1*df_k_1');
      case 'PR'
          beta_k =  max([((df_k)*(df_k - df_k_1)')/(norm(df_k_1)^2), 0]);
    end
    p_k = -df_k + p_k * beta_k;
    % Store iteration info
    info.cost = [info.cost F.f(x_k)];
    info.alphas = [info.alphas alpha_k];
    % Compute relative error norm
    normError = norm(x_k - x_k_1)/norm(x_k_1); 
    % Increment iterations
    nIter = nIter + 1;
end

% Assign output values 
xMin = x_k;
fMin = F.f(x_k); 

