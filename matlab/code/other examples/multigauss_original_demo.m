% Multivariate Bayesian Online Changepoint Detection
% by Ilaria Lauzana

close all;
clearvars;


%% Import data

% 2d toy dataset
% load('2D_Toy_Data.mat');
% X = [Xn{3,1}'];

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

% Preprocessed Peeling dataset
% load('proc_data.mat');
% X = proc_data{1,1}.active.X(1:7,:)'


%% Initialization

% Time steps to generate
T = size(X,1);

% Matrix for posterior probabilities 
R = zeros([T+1 T]);
R(1,1) = 1;                 % changepoint assumed before stream of data 

% Maximum posterior probability for each time step
maxes  = zeros(T+1,1);

% Changepoints found by algorithm
ChPnt = [0];
last_cp = 0;                % time step of last changepoint
prev_cp = 0;                % time step of previous changepoint

% Counters for changepoints and parameters storage
cnt = 0;
flag = 0;
change_values = 0;


%% Specify the hazard function (prior on segment length).

% To use uniform prior (1/lambda):
lambda = 200;
min_len = 0;
hazard_func  = @(r) constant_hazard(r, lambda);

% To use 3p logistic function:
% min_len = 0;
% h = 0.01;
% a = 0.01;
% b = 0;
% hazard_func = @(r) logistic_h(r, [h,a,b]);

% To use truncated gaussian:
% mu_len = 1000;
% sigma_len = 800; 
% min_len = 300;
% hazard_func  = @(r) truncated_gauss(r, mu_len, sigma_len, min_len);

%% Specify model of data and conjugate prior

% Parameters for NIW conjugate prior
dim             = size(X,2);               %dimension of data
mu0             = zeros(1,dim);
kappa0          = 1;
nu0             = dim;
sigma0          = eye(dim);

% Current set of parameters start out at the prior and is updated with data
muT             = mu0;
kappaT          = kappa0;
nuT             = nu0;
sigmaT(:,:,1)   = sigma0;

% Truncated parameters of the current segment to store current mean and
% covariance
muT_segment             = mu0;
kappaT_segment          = kappa0;
nuT_segment             = nu0;
sigmaT_segment(:,:,1)   = sigma0;

% Parameters of all the segments
mu_saved = [];
covar_saved = [];
% Parameters of current segment (then inserted in "saved")
mu = [];
covar = [];


%% Algorithm

tic;

% Loop over the data
for t=1:T
 
  curr_t = t-last_cp;
  
  found = find(CP==t);
  if found
    timecp = tic;
    f = found;
  end
    
  % Predictive distribution for the new datum as NIW
    predprobs = studentpdf_multi(X(t,:), muT, ...
              sigmaT, nuT, kappaT, dim);
 
  % Hazard function for current t
  if t > min_len 
    H = [zeros(min_len,1); hazard_func([min_len+1:t]')];
  else
    H = zeros(t,1);
  end
  
  % Growth probabilities: shift the probabilities down and to the right,
  % scaled by the hazard function and the predictive probabilities
  R(2:t+1,t+1) = R(1:t,t) .* predprobs' .* (1-H);
  
  % Changepoint probability
  R(1,t+1) = sum( R(1:t,t) .* predprobs' .* H );
  
  % Renormalize run length probabilities with model evidence
  R(:,t+1) = R(:,t+1) ./ sum(R(:,t+1));
  
  % Maximum posterior probability
  maxes(t) = find(R(:,t)==max(R(:,t)));
  
  % If the run length drops
  if t > 1 && maxes(t) - maxes(t-1) < -10
     flag = 1;
     % Save current segment's parameters
     mu = muT_segment(curr_t,:);
     covar = 2*(kappaT_segment(curr_t)+1).*sigmaT_segment(:,:,curr_t) ...
         ./ (nuT_segment(curr_t)*kappaT_segment(curr_t));
  % Check that the run length really dropped
  elseif flag == 1 && t>1
     if abs(maxes(t) - maxes(t-1)) < 10
        cnt = cnt + 1;
        if cnt > 10
        % if run length is still low after 10 checks, changepoint found
           change_values = 1;
           flag = 0;
           cnt = 0;
           ChPnt = [ChPnt; t-maxes(t)+1];
           delayCP(f) = toc(timecp);
           prev_cp = last_cp;
           last_cp = ChPnt(length(ChPnt));
           curr_t = t-last_cp;
           % Store parameters of the segment
           mu_saved = [mu_saved; mu];
           covar_saved = [covar_saved; covar];
        end
     else
        flag = 0;
        cnt = 0;
     end
  end
  
  % If changepoint was found, cut the segment's parameters
  if change_values == 1
     muT_segment = muT_segment(1:curr_t,:);
     nuT_segment = nuT_segment(1:curr_t);
     kappaT_segment = kappaT_segment(1:curr_t);
     sigmaT_segment = sigmaT_segment(:,:,1:curr_t);
     change_values = 0;
  end
  
  % Update the parameters for each possible run length
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
  
  % Update the segment parameters for each possible run length
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

% Store last segment's parameters
mu_saved = [mu_saved; muT_segment(curr_t,:)];
covar_saved = [covar_saved; 2*(kappaT_segment(curr_t)+1).*sigmaT_segment(:,:,curr_t)...
    ./(nuT_segment(curr_t)*kappaT_segment(curr_t))];

elapsed = toc;

%% Plot the data with found changepoints

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

CPs   = [ChPnt+1, [ChPnt(2:end); T]];
N_CPs = length(CPs);

figure('Color',[1 1 1])
for i=1:N_CPs
    subplot(N_CPs,1,i)
    hold on
    scatter(X(CPs(i,1):CPs(i,2),1), X(CPs(i,1):CPs(i,2),2), 10, [rand rand rand])
    plot(mu_saved(i,1), mu_saved(i,2), '*')
end