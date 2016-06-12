

function [logH, logmH, dlogH, dlogmH] = logistic_logh(v, theta_h)
% h(t) = h * logistic(at + b)
% theta_h: [logit(h), a, b]
% derived on p. 230 - 232 of DPI notebook

T = size(v, 1);

h = logistic(theta_h(1));
a = theta_h(2);
b = theta_h(3);

% = log(1 - h) = log(1 - logistic(theta_h(1)), but will work for extreme input 
logmh = loglogistic(-theta_h(1));
logh = loglogistic(theta_h(1));

% logisticPos would be: (if it was needed)
% logisticPos = logistic(a .* v + b);
logisticNeg = logistic(-a .* v - b);

logH = loglogistic(a .* v + b) + logh;
% TODO rewrite to avoid Inf-Inf in extreme cases
logmH = logsumexp(-a .* v - b, logmh) + loglogistic(a .* v + b);

% Derivatives
dlogH = zeros(T, 3);
dlogH(:, 1) = logistic(-theta_h(1));
dlogH(:, 2) = logisticNeg .* v;
dlogH(:, 3) = logisticNeg;

dlogmH = zeros(T, 3);
dlogmH(:, 1) = dlogsumexp(-a .* v - b, 0, logmh, -h);
dlogmH(:, 2) = dlogsumexp(-a .* v - b, -v, logmh, 0) + v .* logisticNeg;
dlogmH(:, 3) = dlogsumexp(-a .* v - b, -1, logmh, 0) + logisticNeg;

function logZ = logsumexp(x, c)

maxx = max(x, c);
maxx(~isfinite(maxx)) = 0;
logZ = log(exp(x - maxx) + exp(c - maxx)) + maxx;

function dlogZ = dlogsumexp(x, dx, c, dc)

maxx = max(x, c);
maxx(~isfinite(maxx)) = 0;
% TODO rewrite to avoid 0/0 in extreme cases
dlogZ = (exp(x - maxx) .* dx + exp(c - maxx) .* dc) ./ (exp(x - maxx) + exp(c - maxx));
