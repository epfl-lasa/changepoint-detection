% Demonstration of online detection of a change in 2d Gaussian parameters.
%
% Changes by Ilaria Lauzana

% Start with a clean slate.
close all;
clearvars;


%% Import data

% 2d toy dataset
% load('2D_Toy_Data.mat');
% X = [Xn{4,1}'];

% Carrot Grating dataset
load('CarrotGrating.mat');
X = [Xn{1,1}'];

% Dough Rolling dataset
% load('Rolling_Raw.mat');
% X = [Xn{1,1}'];

% Preprocessed Dough Rolling dataset
% load('Rolling_Processed.mat');
% X = [Xn_ch{3,1}'];

% 30 Industry Portfolios
% load('30_industry.mat');
% X = thirty_industry(:,2:end);

% Bee sequence
% load('bee_seq2.mat');
% X = bee;


%% Initialization

% How many time steps to generate?
T = size(X,1);

% Specify the hazard function.

% To use uniform prior 1/lambda:
lambda = 200;
min_len = 0;
hazard_func  = @(r) constant_hazard(r, lambda);

% To use logistic:
% min_len = 0;
% h = 0.01;
% a = 0.01;
% b = 0;
% hazard_func = @(r) logistic_h(r, [h,a,b]);

% To use truncated gaussian:
% mu_len = 200;
% sigma_len = 150; 
% min_len = 50;
% hazard_func  = @(r) truncated_gauss(r, mu_len, sigma_len, min_len);

% Parameters for NIW conjugate prior
dim = size(X,2);               %dimension of data
mu0 = zeros(1,dim);
kappa0 = 1;
nu0 = dim;
sigma0 = eye(dim);

%% Plot data and real changepoints (for bees)

% figure;
% plot([1:T]', X);
% hold on;
% for l=1:size(bee_change)
%     if bee_change(l) 
%         plot([l l],ylim,'r');
%     end
% end
% grid on;

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

% Truncated parameters of the current segment to store current mean and
% covariance
muT_segment    = mu0;
kappaT_segment = kappa0;
nuT_segment = nu0;
sigmaT_segment(:,:,1) = sigma0;

% Keep track of the maximums.
maxes  = zeros(T+1,1);

% Variables used to keep track of changepoints and segment parameters
ChPnt = [1];
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
  
  %if flag == 0 && maxes(t) < 25 && t>25
  if t > 1 && maxes(t) - maxes(t-1) < -10 %&& maxes(t) < 20
     flag = 1;
     mu = muT_segment(curr_t,:);
     covar = 2*(kappaT_segment(curr_t)+1).*sigmaT_segment(:,:,curr_t) ...
         ./ (nuT_segment(curr_t)*kappaT_segment(curr_t));
  elseif flag == 1 %&& t>1
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
     muT_segment = muT_segment(1:curr_t,:);
     nuT_segment = nuT_segment(1:curr_t);
     kappaT_segment = kappaT_segment(1:curr_t);
     sigmaT_segment = sigmaT_segment(:,:,1:curr_t);
     change_values = 0;
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
  
  muT0    = [mu0; bsxfun(@rdivide, bsxfun(@plus, bsxfun(@times, ...
      kappaT_segment, muT_segment), X(t,:)), (kappaT_segment+1))];
  kappaT0 = [ kappa0 ; kappaT_segment + 1 ];
  nuT0    = [ nu0    ; nuT_segment + 1 ];
  sigmaT0 = sigmaT_segment;
  sigmaT_segment(:,:,1) = sigma0;
  X_mu = bsxfun(@minus, X(t,:), muT_segment);
  for i = 1:curr_t
      X_mu_2 = X_mu(i,:)'*X_mu(i,:);
      sigmaT_segment(:,:,i+1) = sigmaT0(:,:,i) + kappaT_segment(i) ...
          .* X_mu_2./(2*(kappaT_segment(i)+1));
  end 
  muT_segment     = muT0;
  kappaT_segment  = kappaT0;
  nuT_segment     = nuT0;
  
end

mu_saved = [mu_saved; muT_segment(curr_t,:)];
covar_saved = [covar_saved; 2*(kappaT_segment(curr_t)+1).*sigmaT_segment(:,:,curr_t)...
    ./(nuT_segment(curr_t)*kappaT_segment(curr_t))];

toc;

%% Plot the data and we'll have a look.

subplot(2,1,1);
plot([1:T]', X);
hold on;
for l=1:size(ChPnt) 
    plot([ChPnt(l) ChPnt(l)],ylim,'r');
end
grid on;

% Show the log smears and the maximums.
subplot(2,1,2);
colormap(gray());
imagesc(-log(R));
hold on;
plot([1:T+1], maxes, 'r-');
hold off;


%% Checking data with stored changepoints and paramaters

CPs   = [ChPnt, [ChPnt(2:end)+1; T]];
N_CPs = length(CPs);

figure('Color',[1 1 1])
for i=1:N_CPs
    subplot(N_CPs,1,i)
    hold on
    scatter(X(CPs(i,1):CPs(i,2),1), X(CPs(i,1):CPs(i,2),2), 10, [rand rand rand])
    plot(mu_saved(i,1), mu_saved(i,2), '*')
end