function [H, dH] = logistic_h(v, theta_h)
% h(t) = h * logistic(at + b)
% theta_h: [h, a, b]

h = theta_h(1);
a = theta_h(2);
b = theta_h(3); 

lp = logistic(a .* v + b);
lm = logistic(-a .* v - b);
H = h .* lp;

% Derivatives
dH(:,1) = lp;
lp_lm_v = (lp .* lm .* v);
dH(:,2) = h .* lp_lm_v;
dH(:,3) = h .* lp .* lm;
