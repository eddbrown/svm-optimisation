function [alpha_s, info] = lineSearch(F, x_k, p_k, alpha_max, opts)
% LINESEARCH Line Search algorithm satisfying strong Wolfe conditions 
% alpha_s = lineSearch(F, x_k, p_k, alpha_max, opts)
% 
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
% x_k: current iterate
% p_k: descent direction
% alpha_max: maximum step length 
% opts: line search specific option structure with fields
%   - c1: constant in sufficient decrease condition 
%         f(x_k + alpha_k*p_k) > f(x_k) + c1*alpha_k*(df_k'*p_k)
%         Typically chosen small, (default 1e-4)
%   - c2: constant in strong curvature condition 
%         |df(x_k + alpha_k*p_k)'*p_k| <= c2*|df(x_k)'*p_k| 
%
% OUTPUT
% alpha_s: step length
% info: structure containing alpha_j history
%
% Reference: Algorithm 3.5 from Nocedal, Numerical Optimization
%
% It generates a monotonically increasing sequence of step lenghts alpha_j. 
% Uses the fact that interval (alpha_j_1, alpha_j) contains step lengths satisfying strong Wolfe conditions 
% if one of the conditions below is satisfied:
% (C1) alpha_j violates the sufficient decrease condition 
% (C2) phi(alpha_j) >= phi(alpha_j_1)
% (C3) dphi(alpha_j) >= 0
%
% Copyright (C) 2017 Kiko Rullan, Marta M. Betcke 

% Paramters
% Multiple of alpha_j used to generate alpha_{j+1}
FACT = 10; 

% Calculate handle to function phi(alpha) = f(x_k + alpha*p_k)
% Phi: function structure with fields
% - phi: function handler
% - dphi: derivative handler
Phi.phi = @(alpha) F.f(x_k + alpha*p_k);
Phi.dphi = @(alpha) (F.df(x_k + alpha*p_k))*p_k';

% Initialization
alpha(1) = 0;
phi_i(1) = Phi.phi(0);
dphi_i(1) = Phi.dphi(0);
alpha(2) = 0.9*alpha_max; %0.5*alpha_max;
alpha_s = 0;
n = 2;
maxIter = 10;
stop = false;

while (n < maxIter && stop == false)
    phi_i(n) = Phi.phi(alpha(n));
    dphi_i(n) = Phi.dphi(alpha(n));
    if(phi_i(n) > phi_i(1) + opts.c1*alpha(n)*dphi_i(1) || (phi_i(n) >= phi_i(n-1) && n > 2))
        alpha_s = zoomInt(Phi, alpha(n-1), alpha(n), opts.c1, opts.c2);
        stop = true; 
    elseif(abs(dphi_i(n)) <= -opts.c2*dphi_i(1))
        alpha_s = alpha(n);
        stop = true;
    elseif(dphi_i(n) >= 0)
        alpha_s = zoomInt(Phi, alpha(n), alpha(n-1), opts.c1, opts.c2);
        stop = true;
    end;
    %alpha(n+1) = 0.5*(alpha(n)+alpha_max);
    alpha(n+1) = max(FACT*alpha(n), alpha_max);
    n = n + 1;
end

info.alphas = alpha;    
