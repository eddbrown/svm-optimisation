function [alpha, info] = zoomInt(Phi, alpha_l, alpha_h, c1, c2)
% ZOOMINT Zoom algorithm for line search with strong Wolfe conditions
% alpha = zoomInt(Phi, alpha_l, alpha_h, c1, c2)
%
% INPUTS
% Phi: structure for function of step length phi(alpha) = f(x_k + alpha*p_k) with fields
%   - phi: function handler
%   - dphi: derivative handler
% alpha_l: lower boundary of the trial interval
% alpha_h: upper boundary of the trial interval
% c1 & c2: constants for Wolfe conditions (see lineSearch.m)
%
% OUTPUT
% alpha: step length
% info: structures containing iteration history
%
% Reference: Algorithm 3.5 from Nocedal, Numerical Optimization
%
% Properties ensured at each iteration
% (P1) Interval (alpha_l, alpha_h) contains step lengths satisfying strong Wolfe conditions.
% (P2) Among the step lengths generated so far satisfying the sufficient decrease condition
%      alpha_l is the one with smallest phi value
% (P3) alpha_h is chose such that dphi(alpha_l)*(alpha_h - alpha_l) < 0
%
% Copyright (C) 2017 Kiko Rullan, Marta M. Betcke 

% Parameters
% Trial step in {'bisection', 'interp2'}
TRIALSTEP = 'bisection';
tol = 10^-4;

% Structure containing information about the iteration
info.alpha_ls = []; 
info.alpha_hs = []; 
info.alpha_js = [];
info.phi_js = [];
info.dphi_js = [];

n = 1;
stop = false;
maxIter = 10;
while (n < maxIter && stop == false)
    % Find trial step length alpha_j in [alpha_l, aplha_h]
    switch TRIALSTEP
      case 'bisection'
        alpha_j = 0.5*(alpha_h + alpha_l);
      case 'interp2'        
    end
    phi_j = Phi.phi(alpha_j);
    
    % Update info
    info.alpha_ls = [info.alpha_ls alpha_l]; 
    info.alpha_hs = [info.alpha_hs alpha_h];
    info.alpha_js = [info.alpha_js alpha_j];
    info.phi_js = [info.phi_js phi_j];
    
    if abs(alpha_h - alpha_l) < tol
      alpha = alpha_j;
      stop = true;
%       warning('Line search stopped because the interval became to small. Return centre of the interval.')
    end
    
    if (phi_j > Phi.phi(0) + c1*alpha_j*Phi.dphi(0) || Phi.phi(alpha_j) >= Phi.phi(alpha_l))
      % alpha_j does not satisfy sufficient decrease condition -> look for alpha < alpha_j
      % or phi(alpha_j) >= phi(alpha_l) 
      % -> [alpha_l, alpha_j]
      alpha_h = alpha_j;
        
      % Update info  
      info.dphi_js = [info.dphi_js NaN];
      
    else 
       % alpha_j satisfies sufficient decrease condition
        dphi_j = Phi.dphi(alpha_j);
         
        % Update info  
        info.dphi_js = [info.dphi_js dphi_j];
            
        if (abs(dphi_j) <= -c2*Phi.dphi(0))
          % alpha_j satisfies strong curvature condition
            alpha = alpha_j;
            stop = true;
        elseif (dphi_j*(alpha_h - alpha_l) >= 0)
          % alpha_h : dphi(alpha_l)*(alpha_h - alpha_l) < 0
          % alpha_j violates this condition but swapping alpha_l <-> alpha_h will reestablish it
          % -> [alpha_j, alpha_l]
            alpha_h = alpha_l;
        end
        alpha_l = alpha_j;
    end
end
            
