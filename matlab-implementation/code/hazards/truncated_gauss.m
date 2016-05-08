function p = truncated_gauss(r, mu, sigma, min)
  p = normcdf(r,mu,sigma) - normcdf(min,mu,sigma);
  