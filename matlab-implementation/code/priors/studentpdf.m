function p = studentpdf(x, mu, var, nu)
%
% p = studentpdf(x, mu, var, nu)
%
  
  % This form is taken from Kevin Murphy's lecture notes.
  c = exp(gammaln(nu/2 + 0.5) - gammaln(nu/2)) .* (nu.*pi.*var).^(-0.5);
  
  p = c .* (1 + (1./(nu.*var)).*(x-mu).^2).^(-(nu+1)/2);

