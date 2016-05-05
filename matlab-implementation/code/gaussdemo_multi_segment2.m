% Demonstration of online detection of a change in 2d Gaussian parameters.
%
% Changes by Ilaria Lauzana

% Start with a clean slate.
close all;
clearvars;


%% Import data

% 2d toy dataset
% load('2D_Toy_Data.mat');
% X = [Xn{3,1}'; Xn{4,1}'];

% Carrot Grating dataset
load('CarrotGrating.mat');
X = [Xn{2,1}'];

% Dough Rolling dataset
% load('Rolling_Raw.mat');
% X = [Xn{2,1}'];

% Preprocessed Dough Rolling dataset
% load('Rolling_Processed.mat');
% X = [Xn_ch{3,1}'];

%% Initialization

% How many time steps to generate?
T = size(X,1);

% Specify the hazard function.

% To use uniform prior 1/lambda:
lambda = 100;
min_len = 0;
hazard_func  = @(r) constant_hazard(r, lambda);

% To use truncated gaussian:
% mu_len = 1000;
% sigma_len = 800; 
% min_len = 300;
% hazard_func  = @(r) truncated_gauss(r, mu_len, sigma_len, min_len);

% Parameters for NIW conjugate prior
dim = size(X,2);               %dimension of data
mu0 = zeros(1,dim);
kappa0 = 1;
nu0 = dim;
sigma0 = eye(dim);


%% Data already generated/imported

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
ChPnt = [0];
mu_saved = [];
covar_saved = [];
mu = [];
covar = [];
cnt = 0;
flag = 0;
change_values = 0;
prev_cp = 0;
last_cp = 0;

tic;

% Loop over the data like we're seeing it all for the first time.
for t=1:T
 
  curr_t = t-last_cp;
    
  % Evaluate the predictive distribution for the new datum under each of
  % the parameters.  This is the standard thing from Bayesian inference.
    predprobs = studentpdf_multi(X(t,:), muT, ...
              sigmaT, nuT, kappaT, dim);
 
  % Evaluate the hazard function for this interval.
  if curr_t > min_len 
    H = [zeros(min_len,1); hazard_func([min_len+1:curr_t]')];
  else
    H = zeros(curr_t,1);
  end
  
  % Evaluate the growth probabilities - shift the probabilities down and to
  % the right, scaled by the hazard function and the predictive
  % probabilities.
  R(2:curr_t+1,t+1) = R(1:curr_t,t) .* predprobs' .* (1-H);
  
  % Evaluate the probability that there *was* a changepoint and we're
  % accumulating the mass back down at r = 0.
  R(1,t+1) = sum( R(1:curr_t,t) .* predprobs' .* H );
  
  % Renormalize the run length probabilities for improved numerical
  % stability.
  R(:,t+1) = R(:,t+1) ./ sum(R(:,t+1));
  
  
  % Store the maximum, to plot later.
  maxes(t) = find(R(:,t)==max(R(:,t)));
  
  if t > 1 && maxes(t) - maxes(t-1) < -10 && maxes(t) < 20
     flag = 1;
     mu = muT(curr_t,:);
     covar = 2*(kappaT(curr_t)+1).*sigmaT(:,:,curr_t) ...
         ./ (nuT(curr_t)*kappaT(curr_t));
  elseif flag == 1 && t>1
     if abs(maxes(t) - maxes(t-1)) < 5
        cnt = cnt + 1;
        if cnt > 10
           ChPnt = [ChPnt; t-maxes(t)+1];
           prev_cp = last_cp;
           last_cp = ChPnt(length(ChPnt));
           curr_t = t-last_cp;
           mu_saved = [mu_saved; mu];
           covar_saved = [covar_saved; covar];
           change_values = 1;
           flag = 0;
           cnt = 0;
        end
     else
        flag = 0;
        cnt = 0;
     end
  end
  
  if change_values == 1
     muT = muT(1:curr_t,:);
     nuT = nuT(1:curr_t);
     kappaT = kappaT(1:curr_t);
     sigmaT = sigmaT(:,:,1:curr_t);
     change_values = 0;
  end
  
  % Update the parameter sets for each possible run length.
  muT0    = [mu0; bsxfun(@rdivide, bsxfun(@plus, bsxfun(@times, kappaT, muT), X(t,:)), (kappaT+1))];
  kappaT0 = [ kappa0 ; kappaT + 1 ];
  nuT0    = [ nu0    ; nuT + 1 ];
  sigmaT0 = sigmaT;
  sigmaT(:,:,1) = sigma0;
  X_mu = bsxfun(@minus, X(t,:), muT);
  for i = 1:curr_t
      X_mu_2 = X_mu(i,:)'*X_mu(i,:);
      sigmaT(:,:,i+1) = sigmaT0(:,:,i) + kappaT(i).*X_mu_2./(2*(kappaT(i)+1));
  end 
  muT     = muT0;
  kappaT  = kappaT0;
  nuT     = nuT0;
  
end

mu_saved = [mu_saved; muT(curr_t,:)];

toc;

%% Plot the data and we'll have a look.

subplot(3,1,1);
plot([1:T]', X(:,1), 'b-', ChPnt, zeros(size(ChPnt)), 'rx');
grid on;
if dim > 1
    subplot(3,1,2);
    plot([1:T]', X(:,2), 'b-', ChPnt, zeros(size(ChPnt)), 'rx');
    grid on;
end


% Show the log smears and the maximums.
subplot(3,1,3);
colormap(gray());
imagesc(-log(R));
hold on;
plot([1:T+1], maxes, 'r-');
hold off;


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