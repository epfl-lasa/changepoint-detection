n = 2^40;
p = [0.4; 0.6];
x = sample_hist(p, n);
assert(sum(x) == n);
assert(max(abs(x/n - p)) < 1e-4);
