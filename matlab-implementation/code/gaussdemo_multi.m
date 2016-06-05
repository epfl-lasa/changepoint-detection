% Demonstration of online detection of a change in 2d Gaussian parameters.
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
% Changes by Ilaria Lauzana

% First, we wil specify the prior.  We will then generate some fake data
% from the prior specification.  We will then perform inference. Then
% we'll plot some things.

% Start with a clean slate.
close all;
clearvars;

%% Initialization

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

% To use uniform prior 1/lambda:
lambda = 200;
min_len = 0;
hazard_func  = @(r) constant_hazard(r, lambda);

% To use truncated gaussian:

% mu_len = 300;
% sigma_len = 50; 
% min_len = 200;
% hazard_func  = @(r) truncated_gauss(r, mu_len, sigma_len, min_len);

% This data is Gaussian with unknown mean and variance.  We are going to
% use the standard conjugate prior of a normal-inverse-gamma.  Note that
% one cannot use non-informative priors for changepoint detection in
% this construction.  The NIG yields a closed-form predictive
% distribution, which makes it easy to use in this context.  There are
% lots of references out there for doing this kind of inference - for
% example Chris Bishop's "Pattern Recognition and Machine Learning" in
% Chapter 2.  Also, Kevin Murphy's lecture notes.
dim = 6 ;               %dimension of data
mu0    = zeros(1,dim);
kappa0 = 1;
nu0 = dim;
sigma0 = eye(dim);


%% Generate data

% This will hold the data.  Preallocate for a slight speed improvement.
X = zeros([T dim]);

% Store the times of changepoints.  It's useful to see them.
CP = [0];

% Generate the initial parameters of the Gaussian from the prior.
curr_ivar = wishrnd(inv(sigma0),nu0);
curr_mean = mvnrnd(mu0,curr_ivar);

% The initial run length is zero.
curr_run = 0;

% Now, loop forward in time and generate data.
for t=1:T
  
  % Get the probability of a new changepoint.
  p = hazard_func(curr_run);
      
  % Randomly generate a changepoint, perhaps.
  if rand() < p 
      if curr_run > 200
          % Generate new Gaussian parameters from the prior.
          curr_ivar = wishrnd(inv(sigma0),nu0);
          curr_mean = mvnrnd(mu0,curr_ivar);

          % The run length drops back to zero.
          curr_run = 0;
          
          % Add this changepoint to the end of the list.
          CP = [CP ; t-1];
      end
  else
    
    % Increment the run length if there was no changepoint.
    curr_run = curr_run + 1;
  end
  
  % Draw data from the current parameters.
  X(t,:) = mvnrnd(curr_mean,curr_ivar);
end

%% Data already generated/imported

%Plot the data and we'll have a look.
subplot(3,1,1);
plot([1:T]', X(:,1), 'b-');%, CP, zeros(size(CP)), 'rx');
grid on;
if dim > 1
    subplot(3,1,2);
    plot([1:T]', X(:,2), 'b-');%, CP, zeros(size(CP)), 'rx');
    grid on;
end

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
nuT = nu0;
sigmaT(:,:,1) = sigma0;

% Keep track of the maximums.
maxes  = zeros(T+1,1);
ChPnt = [];
mu_saved = [];
mu = [];
cnt = 0;
flag = 1;

tic;

% Loop over the data like we're seeing it all for the first time.
for t=1:T
 
  % Evaluate the predictive distribution for the new datum under each of
  % the parameters.  This is the standard thing from Bayesian inference.
    predprobs = studentpdf_multi(X(t,:), muT, ...
              sigmaT, nuT, kappaT, dim);
 
  % Evaluate the hazard function for this interval.
  if t > min_len 
    H = [zeros(min_len,1); hazard_func([min_len+1:t]')];
  else
    H = zeros(t,1);
  end
  
  % Evaluate the growth probabilities - shift the probabilities down and to
  % the right, scaled by the hazard function and the predictive
  % probabilities.
  R(2:t+1,t+1) = R(1:t,t) .* predprobs' .* (1-H);
  
  % Evaluate the probability that there *was* a changepoint and we're
  % accumulating the mass back down at r = 0.
  R(1,t+1) = sum( R(1:t,t) .* predprobs' .* H );
  
  % Renormalize the run length probabilities for improved numerical
  % stability.
  R(:,t+1) = R(:,t+1) ./ sum(R(:,t+1));

  
  % Store the maximum, to plot later.
  maxes(t) = find(R(:,t)==max(R(:,t)));
  
  if flag == 0 && maxes(t) < 15
     flag = 1;
     mu = muT(t,:);
  elseif flag == 1 && t>1
     if maxes(t) - maxes(t-1) < 5
        cnt = cnt + 1;
        if cnt > 15
           ChPnt = [ChPnt; t-maxes(t)];
           mu_saved = [mu_saved; mu];
           flag = 0;
           cnt = 0;
        end
     else
        flag = 0;
        cnt = 0;
     end
  end
  
  
  % Update the parameter sets for each possible run length.
  muT0    = [mu0; bsxfun(@rdivide, bsxfun(@plus, bsxfun(@times, kappaT, muT), X(t,:)), (kappaT+1))];
  kappaT0 = [ kappa0 ; kappaT + 1 ];
  nuT0    = [ nu0    ; nuT + 1 ];
  sigmaT0 = sigmaT;
  sigmaT(:,:,1) = sigma0;
  X_mu = bsxfun(@minus, X(t,:), muT);
  for i = 1:t
      X_mu_2 = X_mu(i,:)'*X_mu(i,:);
      sigmaT(:,:,i+1) = sigmaT0(:,:,i) + kappaT(i).*X_mu_2./(2*(kappaT(i)+1));
  end 
  muT     = muT0;
  kappaT  = kappaT0;
  nuT     = nuT0;
  
end

mu_saved = [mu_saved; muT(T,:)];

toc;

% Show the log smears and the maximums.
subplot(3,1,3);
colormap(gray());
imagesc(-log(R));
hold on;
plot([1:T+1], maxes, 'r-');
hold off;


%% Checking Data

CPs   = [CP+1 , [CP(2:end)+1; T]];
N_CPs = length(CPs);

figure('Color',[1 1 1])
for i=1:N_CPs
    subplot(N_CPs,1,i)
    hold on
    scatter(X(CPs(i,1):CPs(i,2),1), X(CPs(i,1):CPs(i,2),2), 10, [rand rand rand])
    plot(muT(CPs(i,2)+1,1), muT(CPs(i,2)+1,2), '*')
end

%% Checking data with stored changepoints and paramaters

CPs   = [ChPnt+1, [ChPnt(2:end)+2; T]];
N_CPs = length(CPs);

figure('Color',[1 1 1])
for i=1:N_CPs
    subplot(N_CPs,1,i)
    hold on
    scatter(X(CPs(i,1):CPs(i,2),1), X(CPs(i,1):CPs(i,2),2), 10, [rand rand rand])
    plot(mu_saved(i,1), mu_saved(i,2), '*')
end