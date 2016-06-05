Code for "Bayesian Online Changepoint Detection" by Adams and MacKay.
Description found here:
http://hips.seas.harvard.edu/content/bayesian-online-changepoint-detection

Lightspeed toolbox: dependency needed to run the univariate code.
To install, enter the "lightspeed" folder and run "install_lightspeed" from Matlab.

Univariate version: run "gaussdemo.m", found in folder "code".

Multivariate version:
	run "code/multi_gauss.m" for version considering all run lengths
	run "code/multi_gauss_fast.m" for new version, considering only run lengths starting from last changepoint