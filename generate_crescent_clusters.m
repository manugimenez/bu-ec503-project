% Daniel Birch
% dbirch@bu.edu

% Clean up
clc;
clear all;
close all;
rng('shuffle');

% User-defined parameters
numberOfClusters = 10;
nPoints = 1000; % Number of points
d = 2; % Number of dimensions (features)
radiusScale = pi/8;% Length scale of an individual cluster
allowCovariances = true;

% Assign random probabilities to each cluster
p = exp(randn(numberOfClusters, 1));%log-normal probabilities
p = p / sum(p);%Normalize

% Generate random means covariances for each class
mu = 2 * pi * rand(numberOfClusters, 1);
sigma = radiusScale * exp(randn(numberOfClusters, d)) / exp(0.5);

% covMatrices = NaN(numberOfClusters, d, d);
% for n = 1:numberOfClusters
%     if allowCovariances
%         [Q,~] = qr(randn(d,d));% Generate an orthogonal matrix from the 
%                                % Haar distribution
%         C = transpose(Q)*diag(sigma(n,:))*Q;% Generate the covariance matrix                     
%     else
%         C = diag(sigma(n,:));
%     end
%     
%     covMatrices(n,:,:) = C;
% end


% Allocate space
y = NaN(nPoints, 1);


% Main loop
for n = 1:nPoints
    r = rand(1);
    
    thisClass = find( r < cumsum(p) , 1, 'first');
    theta = mu(thisClass) + sigma(thisClass) * rand(1);
    x(n,:) = [cos(theta), sin(theta)];
    y(n) = thisClass;
end

% Plot the data
if (d == 2)
    figure;
    plot(cos(mu), sin(mu), 'ko');
    hold on;
    h = gscatter(x(:,1), x(:,2), y, [], [], [], 'off');
    xlabel('{\itx}_1');
    ylabel('{\itx_2}');
    xlim([-1.01, 1.01]);
    ylim([-1.01, 1.01]);
    axis equal;
end

