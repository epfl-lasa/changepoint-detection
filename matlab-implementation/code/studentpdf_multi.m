function p = studentpdf_multi(x, mu, covar, nu, kappa, dim)
% Now covar is a (t x dim x dim) matrix and mu is a (t x dim) matrix
  
  % This form is taken from Kevin Murphy's lecture notes.
%   c = exp(gammaln(nu/2 + 0.5) - gammaln(nu/2)) .* (nu.*pi.*var).^(-0.5);
%   
%   p = c .* (1 + (x-mu)'*(1./(nu.*var))*(x-mu)).^(-(nu+1)/2);
  
% x: 1xd vector
% mu: nxd matrix 
% covar: dxdxn -> dxd symmetric, positive definite matrix
% nu: positive scalar


X_mu = bsxfun(@minus, x, mu);
m = size(X_mu,1);
for i = 1:m
    %mult(i) = X_mu(i,:)*covar((2*i-1):2*i,:)*X_mu(i,:)';
    c = 2*(kappa(i)+1).*inv(covar(:,:,i))./((nu(i)).*kappa(i));
    mult = X_mu(i,:)*c*X_mu(i,:)';
    logc = gammaln(nu(i)/2 + dim/2) - gammaln(nu(i)/2) ...
        + 0.5*log(det(c)) ...
        - (dim/2)*log(nu(i)) - (dim/2)*log(pi);
    p(i) = logc - ((nu(i)+dim)/2)*log1p(mult/nu(i));
    %p(i) = exp(logc - ((nu(i)+dim)/2)*log1p(mult/nu(i)));
    %p(i) = exp(logc)*(exp(log1p(mult/nu(i)))^(-(nu(i)+dim)/2));
    if p(i) > 0
       disp('prob > 1 !') 
    end
end
