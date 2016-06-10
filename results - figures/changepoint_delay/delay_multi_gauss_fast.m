
% Start with a clean slate.
close all;
clearvars;

%% Initialization

% How many time steps to generate?
T = 1000;
% How many dimensions?
dim = 10;               %dimension of data

% Specify the hazard function:
lambda = 200;
min_len = 0;
hazard_func  = @(r) constant_hazard(r, lambda);

% Specify prior:
mu0    = zeros(1,dim);
kappa0 = 1;
nu0 = dim;
sigma0 = eye(dim);


%% Generate data

% This will hold the data.  Preallocate for a slight speed improvement.
X = zeros([T dim]);

% Store the times of changepoints:
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

% Matrix for posterior probabilities 
R = zeros([T+1 T]);
R(1,1) = 1;                 % changepoint assumed before stream of data 

% Maximum posterior probability for each time step
maxes  = zeros(T+1,1);

% Changepoints found by algorithm
ChPnt = [0];
last_cp = 0;                % time step of last changepoint
prev_cp = 0;                % time step of previous changepoint
delayCP = zeros(length(CP),1);

% Counters for changepoints and parameters storage
cnt = 0;
flag = 0;
change_values = 0;


% Specify model of data and conjugate prior

% Current set of parameters start out at the prior and is updated with data
muT             = mu0;
kappaT          = kappa0;
nuT             = nu0;
sigmaT(:,:,1)   = sigma0;

% Parameters of all the segments
mu_saved = [];
covar_saved = [];
% Parameters of current segment (then inserted in "saved")
mu = [];
covar = [];


% Algorithm

initial = tic;

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
  if curr_t > min_len 
    H = [zeros(min_len,1); hazard_func([min_len+1:curr_t]')];
  else
    H = zeros(curr_t,1);
  end
  
  % Growth probabilities: shift the probabilities down and to the right,
  % scaled by the hazard function and the predictive probabilities
  R(2:curr_t+1,t+1) = R(1:curr_t,t) .* predprobs' .* (1-H);
  
  % Changepoint probability
  R(1,t+1) = sum( R(1:curr_t,t) .* predprobs' .* H );
  
  % Renormalize run length probabilities with model evidence
  R(:,t+1) = R(:,t+1) ./ sum(R(:,t+1));
  
  % Maximum posterior probability
  maxes(t) = find(R(:,t)==max(R(:,t)));
  
  % If the run length drops
  if t > 1 && maxes(t) - maxes(t-1) < -10 % && maxes(t) < 20
     flag = 1;
     % Save current segment's parameters
     mu = muT(curr_t,:);
     covar = 2*(kappaT(curr_t)+1).*sigmaT(:,:,curr_t) ...
         ./ (nuT(curr_t)*kappaT(curr_t));
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
  
  % If changepoint was found, cut the run length
  if change_values == 1
     muT = muT(1:curr_t,:);
     nuT = nuT(1:curr_t);
     kappaT = kappaT(1:curr_t);
     sigmaT = sigmaT(:,:,1:curr_t);
     change_values = 0;
  end
  
  % Update the parameter sets for each possible run length
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

% Store last segment's parameters
mu_saved = [mu_saved; muT(curr_t,:)];
covar = 2*(kappaT(curr_t)+1).*sigmaT(:,:,curr_t) ...
         ./ (nuT(curr_t)*kappaT(curr_t));
covar_saved = [covar_saved; covar];


elapsed = toc(initial);
% elapsed(k) = toc;
% nCP(k) = size(ChPnt,1);

% end


%% Plot the time and changepoints

% figure;
% subplot(2,1,1);
% title('Time comparison between full and fast version');
% errorbar(mean(elapsed,2), std(elapsed,0,2), 'b-*');
% hold on;
% grid on;

% Show the log smears and the maximums.
% title('Number of changepoints found');
% subplot(2,1,2);
% hold on;
% plot(nCP, 'b*');
% grid on;