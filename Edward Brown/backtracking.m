function [alpha, info] = backtracking(F, x_k, p, alpha0, opts)
% BACKTRACKING Backtracking line search to satisfy sufficient decrease condition 
% [alpha, info] = backtracking(F, x_k, p, alpha0, opts)
%
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
% x_k: current iterate
% p: descent direction
% alpha0: initial step length 
% opts: backtracking specific option structure with fields
%   - rho: in (0,1) backtraking step length reduction factor
%   - c1: constant in sufficient decrease condition f(x_k + alpha_k*p) > f(x_k) + c1*alpha_k*(df_k'*p)
%         Typically chosen small, (default 1e-4). 
%
% OUTPUTS
% alpha: step length
% nIter: number of iterations 
% info: structure with information about the backtracking iteration 
%   - alphas: step lengths history 
%
% Copyright (C) 2017  Marta M. Betcke, Kiko Rullan


% Default values
if nargin < 5 || ~isfield(opts, 'c1')
  opts.c1 = 1e-4;
end
% Choose 
% rho = 0.1 for steepest descent, conjugate gradients
% rho = 0.9 for Newton, Quasi-Newton
if nargin < 5 || ~isfield(opts, 'rho')
  opts.rho = 0.9;  % Newton
end
if nargin < 4 
  alpha0 = 1; % Newton
end

% Initialize info structure
info.alphas = alpha0;
info.rho = opts.rho;
info.c1 = opts.c1;

% Initial step length
alpha = alpha0; 

% Compute f, grad f at x_k
f_k = F.f(x_k);
df_k = F.df(x_k);

% Backtracking linesearch for computing step length
while F.f(x_k + alpha*p) > f_k + opts.c1*alpha*(df_k')*p
  alpha = opts.rho*alpha;
  info.alphas = [info.alphas alpha];
end
