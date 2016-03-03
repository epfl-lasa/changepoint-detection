% Demonstration of online detection of a change in 1d Gaussian parameters.
%
% Implementation of:
% @TECHREPORT{ adams-mackay-2007,
%    AUTHOR = {Ryan Prescott Adams and David J.C. MacKay},
%    TITLE  = "{B}ayesian Online Changepoint Detection",
%    INSTITUTION = "University of Cambridge",
%    ADDRESS = "Cambridge, UK",
%    YEAR = "2007",
%    NOTE = "arXiv:0710.3742v1 [stat.ML]"
% }
%
% Thanks to Ryan Turner and Miguel Lazaro Gredilla for pointing out bugs
% in this.

% First, we wil specify the prior.  We will then generate some fake data
% from the prior specification.  We will then perform inference. Then
% we'll plot some things.

% Start with a clean slate.
clear;

% How many time steps to generate?
T = 1000;

% Specify the hazard function.
% This is a handle to a function that takes one argument - the number of
% time increments since the last changepoint - and returns a value in
% the interval [0,1] that is the probability of changepoint.  Generally
% you might want to have your hazard function take parameters, so using
% an anonymous function is helpful.  We're going to just use the simple
% constant-rate hazard function that gives geomtrically-drawn intervals
% between changepoints.  We'll specify the rate via a mean.
lambda        = 200;
hazard_func  = @(r) constant_hazard(r, lambda);

% This data is Gaussian with unknown mean and variance.  We are going to
% use the standard conjugate prior of a normal-inverse-gamma.  Note that
% one cannot use non-informative priors for changepoint detection in
% this construction.  The NIG yields a closed-form predictive
% distribution, which makes it easy to use in this context.  There are
% lots of references out there for doing this kind of inference - for
% example Chris Bishop's "Pattern Recognition and Machine Learning" in
% Chapter 2.  Also, Kevin Murphy's lecture notes.
mu0    = 0;
kappa0 = 1;
alpha0 = 1;
beta0  = 1;

% This will hold the data.  Preallocate for a slight speed improvement.
X = zeros([T 1]);

% Store the times of changepoints.  It's useful to see them.
CP = [0];

% Generate the initial parameters of the Gaussian from the prior.
curr_ivar = randgamma(alpha0)/beta0;
curr_mean = (kappa0*curr_ivar)^(-0.5)*randn() + mu0;

% The initial run length is zero.
curr_run = 0;

% Now, loop forward in time and generate data.
for t=1:T
  
  % Get the probability of a new changepoint.
  p = hazard_func(curr_run);
  
  % Randomly generate a changepoint, perhaps.
  if rand() < p
    
    % Generate new Gaussian parameters from the prior.
    curr_ivar = randgamma(alpha0)*beta0;
    curr_mean = (kappa0*curr_ivar)^(-0.5)*randn() + mu0;

    % The run length drops back to zero.
    curr_run = 0;
    
    % Add this changepoint to the end of the list.
    CP = [CP ; t];
    
  else
    
    % Increment the run length if there was no changepoint.
    curr_run = curr_run + 1;
  end
  
  % Draw data from the current parameters.
  X(t) = curr_ivar^(-0.5) * randn() + curr_mean;
end

% Plot the data and we'll have a look.
subplot(2,1,1);
plot([1:T]', X, 'b-', CP, zeros(size(CP)), 'rx');
grid;

% Now we have some data in X and it's time to perform inference.
% First, setup the matrix that will hold our beliefs about the current
% run lengths.  We'll initialize it all to zero at first.  Obviously
% we're assuming here that we know how long we're going to do the
% inference.  You can imagine other data structures that don't make that
% assumption (e.g. linked lists).  We're doing this because it's easy.
R = zeros([T+1 T]);

% At time t=1, we actually have complete knowledge about the run
% length.  It is definitely zero.  See the paper for other possible
% boundary conditions.
R(1,1) = 1;

% Track the current set of parameters.  These start out at the prior and
% accumulate data as we proceed.
muT    = mu0;
kappaT = kappa0;
alphaT = alpha0;
betaT  = beta0;

% Keep track of the maximums.
maxes  = zeros([T+1]);

% Loop over the data like we're seeing it all for the first time.
for t=1:T
  
  % Evaluate the predictive distribution for the new datum under each of
  % the parameters.  This is the standard thing from Bayesian inference.
  predprobs = studentpdf(X(t), muT, ...
                         betaT.*(kappaT+1)./(alphaT.*kappaT), ...
                         2 * alphaT);
  
  % Evaluate the hazard function for this interval.
  H = hazard_func([1:t]');
  
  % Evaluate the growth probabilities - shift the probabilities down and to
  % the right, scaled by the hazard function and the predictive
  % probabilities.
  R(2:t+1,t+1) = R(1:t,t) .* predprobs .* (1-H);
  
  % Evaluate the probability that there *was* a changepoint and we're
  % accumulating the mass back down at r = 0.
  R(1,t+1) = sum( R(1:t,t) .* predprobs .* H );
  
  % Renormalize the run length probabilities for improved numerical
  % stability.
  R(:,t+1) = R(:,t+1) ./ sum(R(:,t+1));

  % Update the parameter sets for each possible run length.
  muT0    = [ mu0    ; (kappaT.*muT + X(t)) ./ (kappaT+1) ];
  kappaT0 = [ kappa0 ; kappaT + 1 ];
  alphaT0 = [ alpha0 ; alphaT + 0.5 ];
  betaT0  = [ beta0  ; betaT + (kappaT .*(X(t)-muT).^2)./(2*(kappaT+1)) ];
  muT     = muT0;
  kappaT  = kappaT0;
  alphaT  = alphaT0;
  betaT   = betaT0;
  
  % Store the maximum, to plot later.
  maxes(t) = find(R(:,t)==max(R(:,t)));
  
end

% Show the log smears and the maximums.
subplot(2,1,2);
colormap(gray());
imagesc(-log(R));
hold on;
plot([1:T+1], maxes, 'r-');
hold off;


% Use exportfig to save the image.  You might not have this installed.
if 1
  exportfig(gcf, 'gaussdemo.png', ...
            'Format',     'png', ...
            'Width',      8,   ...
            'Height',     8,   ...
            'FontMode',   'fixed', ...
            'FontSize',   10, ...
            'LineMode',   'fixed', ...
            'LineWidth',  0.5, ...
            'Color',      'rgb', ...
            'Bounds',     'loose');
end
