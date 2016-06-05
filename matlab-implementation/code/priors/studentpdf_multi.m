function p = studentpdf_multi(x, mu, covar, nu, kappa, dim)
% Now covar is a (t x dim x dim) matrix and mu is a (t x dim) matrix
  
  % This form is taken from Kevin Murphy's lecture notes.
%   c = exp(gammaln(nu/2 + 0.5) - gammaln(nu/2)) .* (nu.*pi.*var).^(-0.5);
%   
%   p = c .* (1 + (x-mu)'*(1./(nu.*var))*(x-mu)).^(-(nu+1)/2);
  
% x: 1xd vector
% mu: nxd matrix 
% covar: dxdxn -> dxd symmetric, positive definite matrix
% nu: nx1 vector
% kappa: nx1 vector
% dim: scalar -> # dimensions


X_mu = bsxfun(@minus, x, mu);
m = size(X_mu,1);
for i = 1:m
    c = inv(2*(kappa(i)+1)*covar(:,:,i)/(nu(i)*kappa(i)));
    mult = X_mu(i,:)*c*X_mu(i,:)';
    logc = gammaln(nu(i)/2 + dim/2) - gammaln(nu(i)/2) ...
        + 0.5*log(det(c)) - (dim/2)*log(nu(i).*pi);
    p(i) = exp(logc - ((nu(i)+dim)/2)*log1p(mult/nu(i)));
    
end

