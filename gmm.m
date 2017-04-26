% Daniel Birch
% dbirch@bu.edu

% Clean up
clc;
clear all;
close all;
rng('shuffle');

% User-defined parameters
M = 4;% Number of clusters
dataFile = 'gaussian_clusters_2017_04_25_18_09_11.mat';

% Load the data
A = load(dataFile);
d = A.d; % Number of dimensions
radiusScale = A.radiusScale; % Length scale of a cluster
allowCovariances = A.allowCovariances;
N = A.nPoints;% Number of data points
x = A.x; % Data points

% Initialize the parameter estimates
pHat = exp(randn(M, 1));%log-normal probabilities
pHat = pHat / sum(pHat);%Normalize

% Generate random means covariances for each class
muHat = rand(M, d); %exp(randn(numberOfClusters, d));
sigmaHat = radiusScale * exp(randn(M, d)) / exp(0.5);

covMatricesHat = NaN(M, d, d);
for j = 1:M
    if allowCovariances
        [Q,~] = qr(randn(d,d));% Generate an orthogonal matrix from the 
                               % Haar distribution
        C = transpose(Q)*diag(sigmaHat(j,:))*Q;% Generate the covariance matrix                     
    else
        C = diag(sigma(j,:));
    end
    
    covMatricesHat(j,:,:) = C;
end

%%%%%% Main loop
covJ = NaN(d,d);% The jth covariance matrix estimate
detJ = NaN(1);% Determinant of the jth covariance matrix estimate
a = NaN(N,M);
w = NaN(N,M);

maxIter = 10000;
pp = NaN(M,maxIter+1);
pp(:,1) = pHat;

for iter = 1:maxIter
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

% Plot the probabilities as a function of iteration
figure('name', 'Class probabilities');
plot(pp');
xlim([0, iter]);
xlabel('iteration');
ylabel('probability');
title('Convergence of the EM method');


% Classify the data points
for j = 1:M % Loop over classes
    covJ = squeeze(covMatricesHat(j,:,:));
    detJ = det(covJ);
    xTilde = bsxfun(@minus, x, muHat(j,:));
    
    a(:,j) = exp(-0.5*dot(xTilde/covJ, xTilde, 2)) ...
        / sqrt((2*pi)^M * detJ);
end
[dummy, yHat] = max(a,[],2);

% Build the clusters cell array
clustersHat = cell(M,1);
for j = 1:M
    clustersHat{j} = x(yHat == j, :);
end




% Check the results
p = A.p;
mu = A.mu;
covMatrices = A.covMatrices;
y = A.y;

% Compare the probabilities of each class to the actual probabilities:
if (A.numberOfClusters == M )
    fprintf('\tProbabilities\n');
    fprintf('Class\tp\tpHat\n');
    for j = 1:M
        fprintf('%d\t%.3f\t%.3f\n', j, p(j), pHat(j));
    end



    
    fprintf('\n\n');
    disp('Confusion matrix');
    disp(confusionmat(y, yHat));
end

% Plot the data
if (d == 2)
    x1 = linspace(min(x(:,1)), max(x(:,1)));
    x2 = linspace(min(x(:,2)), max(x(:,2)));
    [xx, yy] = meshgrid(x1, x2);%y here is the y-axis on the graph, NOT the
    %class/cluster id
    
    figure('name', 'Clusters in 2-D');
    plot(mu(:,1), mu(:,2), 'ko');
    hold on;
    gscatter(x(:,1), x(:,2), y);
    plot(muHat(:,1), muHat(:,2), 'kp');
    
    for j = 1:M
        covJinv = inv(squeeze(covMatricesHat(j,:,:)));
        xTilde = xx - muHat(j,1);
        yTilde = yy - muHat(j,2);
        %y here is the y-axis on the graph, NOT the
        %class/cluster id
        dummy = exp(-0.5*(covJinv(1,1) * xTilde.^2 + ...
            2*covJinv(1,2)*xTilde.*yTilde + covJinv(2,2) * yTilde.^2));
        contour(xx, yy, dummy, exp(-1/2), 'k-');
    end
    
    xlabel('{\itx}_1');
    ylabel('{\itx_2}');
    figName = dataFile;
    figName(figName == '_') = ' ';
    title(figName);
end
