% Daniel Birch
% dbirch@bu.edu

% Class ids are sorted by increasing frequency
% Clean up
clc;
clear all;
close all;
rng('shuffle');

% User-defined parameters
numberOfClusters = 4;
nPoints = 1000; % Number of points
d = 2; % Number of dimensions (features)
radiusScale = 0.05;% Length scale of an individual cluster
allowCovariances = true;
saveData = true;

% Generate the name for the saved data
if (saveData)
    fileName = ['gaussian_clusters_', datestr(now, 'yyyy_mm_dd_HH_MM_SS')];
end

% Assign random probabilities to each cluster
p = exp(randn(numberOfClusters, 1));%log-normal probabilities
p = p / sum(p);%Normalize
p = sort(p);

% Generate random means covariances for each class
mu = rand(numberOfClusters, d); %exp(randn(numberOfClusters, d));
sigma = radiusScale * exp(randn(numberOfClusters, d)) / exp(0.5);

covMatrices = NaN(numberOfClusters, d, d);
for n = 1:numberOfClusters
    if allowCovariances
        [Q,~] = qr(randn(d,d));% Generate an orthogonal matrix from the 
                               % Haar distribution
        C = transpose(Q)*diag(sigma(n,:))*Q;% Generate the covariance matrix                     
    else
        C = diag(sigma(n,:));
    end
    
    covMatrices(n,:,:) = C;
end


% Allocate space
y = NaN(nPoints, 1);
x = randn(nPoints, d);

% Main loop
for n = 1:nPoints
    r = rand(1);
    
    thisClass = find( r < cumsum(p) , 1, 'first');
    x(n,:) = x(n,:) * squeeze(covMatrices(thisClass,:,:)) + mu(thisClass, :);
    y(n) = thisClass;
end

% Plot the data
if (d == 2)
    figure;
    plot(mu(:,1), mu(:,2), 'ko');
    hold on;
    gscatter(x(:,1), x(:,2), y);
    xlabel('{\itx}_1');
    ylabel('{\itx_2}');
end

% Put the data into cell arrays
clusters = cell(numberOfClusters, 1);
for thisK = 1:numberOfClusters
    clusters{thisK} = x(y == thisK, :);
end


if saveData
    save(fileName);
    disp(['Data saved as ', fileName]);
end
