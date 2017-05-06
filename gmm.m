function out = gmm(x, k, varargin)
%function out = gmm(x, k, allowCovariances)
% Daniel Birch
% dbirch@bu.edu
% EC 503
% Spring 2017
%
% This script implements the EM algorithm (Dempster, 1977) for the
% Gaussian Mixture Model with a prescribed number of components.
% The K-Means++ algorithm is used for initialization
% (David Arthur and Sergei Vassilvitskii, 2007)
% x is the data (number of data points x number of dimensions)
% k is the number of clusters to form.
% allowCovariances is optional (the default is true).

N = size(x, 1); % Number of data points
d = size(x, 2); % Number of dimensions
M = k; % number of clusters to form

if nargin == 2
    allowCovariances = true;
else
    allowCovariances = varargin{1};
end

%%%%%% Initialize the parameter estimates
% First initialize the means using k-means++
muHat = NaN(M, d);
muHat(1,:) = x(randi(N),:);

figure('name', 'k-means++ initialization');
plot(x(:,1), x(:,2), 'k.');
hold on;
xlabel('x1');
ylabel('x2');

for thisM = 2:M
    q = pdist2(x, muHat(1:thisM-1,:));
    q = min(q,[],2).^2;%Weight by the minimum squared distance
    q = q / sum(q);% Normalize
    q = cumsum(q);
    I = find(rand(1) < q, 1 );
    muHat(thisM,:) = x(I,:);
    plot(x(I,1), x(I,2), 'r.', 'markersize', 20);
    pause(0.2);
end

% Now initialize proportions and covariances
pHat = (1/M) * ones(M,1);% Mixing proportions.  Start with a uniform distribution
sigmaHat = std(x(:)) * ones(M,d);% Covariances

covMatricesHat = NaN(M, d, d);
for j = 1:M
    if allowCovariances
        [Q,~] = qr(randn(d,d));% Generate an orthogonal matrix from the
        % Haar distribution
        C = transpose(Q)*diag(sigmaHat(j,:))*Q;% Generate the covariance matrix
    else
        C = diag(sigmaHat(j,:));
    end
    
    covMatricesHat(j,:,:) = C;
end

%%%%%% Allocate storage
a = NaN(N,M);

maxIter = 10000;
pp = NaN(M,maxIter+1);
pp(:,1) = pHat;

for iter = 1:maxIter %%%%%%%%%%%%%%%%   MAIN LOOP
    % Expectation step
    for j = 1:M % Loop over classes
        covJ = squeeze(covMatricesHat(j,:,:));
        detJ = det(covJ);
        xTilde = bsxfun(@minus, x, muHat(j,:));
        
        a(:,j) = exp(-0.5*dot(xTilde/covJ, xTilde, 2)) ...
            / sqrt((2*pi)^M * detJ);
    end
    
    % Maximization step
    w = bsxfun(@times, a, pHat');
    w = bsxfun(@rdivide, w, sum(w, 2));
    sw = sum(w);
    
    for j = 1:M
        pHat(j) = sw(j) / N;
        muHat(j,:) = transpose(w(:,j)) * x / sw(j);
        
        xTilde = bsxfun(@minus, x, muHat(j,:));
        covMatricesHat(j,:,:) = ...
            transpose(bsxfun(@times, xTilde, w(:,j))) * xTilde / sw(j);
    end
    
    pp(:, iter+1) = pHat;
    
    if ( max(abs(pHat - pp(:,iter))) < eps )
        break;
    end
end

% Sort the results
[pHat, I] = sort(pHat);
muHat = muHat(I,:);
covMatricesHat = covMatricesHat(I,:,:);

% Classify the data points
for j = 1:M % Loop over classes
    covJ = squeeze(covMatricesHat(j,:,:));
    detJ = det(covJ);
    xTilde = bsxfun(@minus, x, muHat(j,:));
    
    a(:,j) = pHat(j) * exp(-0.5*dot(xTilde/covJ, xTilde, 2)) ...
        / sqrt((2*pi)^M * detJ);
end
[~, yHat] = max(a,[],2);

% Build the clusters cell array
clustersHat = cell(M,1);
for j = 1:M
    clustersHat{j} = x(yHat == j, :);
end

out.pHat = pHat;
out.muHat = muHat;
out.yHat = yHat;
out.covMatricesHat = covMatricesHat;
out.pp = pp;
out.iter = iter; %Number of iterations required
out.clustersStructure = clustersHat;%Array of structures for metrics
end