

function [logg, logmG, dlogg, dlogmG] = logistic_logg(v, theta)

[logH, logmH, dlogH, dlogmH] = logistic_logh((1:v)', theta);

logg = zeros(v, 1);
dlogg = zeros(size(dlogH));
for ii = 1:v
  logg(ii) = sum(logmH(1:ii - 1)) + logH(ii);
  dlogg(ii, :) = sum(dlogmH(1:ii - 1, :), 1) + dlogH(ii, :);
end

% exp(logmG) = 1 - G = 1 - cumsum(g) = 1 - cumsum(exp(logg)), but this is a much
% more numerically stable way to do it that won't underflow for G close to 1. 
logmG = cumsum(logmH);
dlogmG = cumsum(dlogmH, 1);
