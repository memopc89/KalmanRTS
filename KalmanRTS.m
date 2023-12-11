function [coordKF, coordRTS] = KalmanRTS(c, R, a, dt)
% By Guillermo Pérez Castro (2020), use Matlab 2020 or earlier
% Position smoothing algorithm for bicycle trajectories based on tracked position measurements (x, y, z) 
% Method: Kalman Filter (KF) and the Rauch, Tung and Striebel smoother algorithm (RTS)
%   INPUT:
%   - c        :   measurement matrix [coord_x coord_y coord_z] (m)
%   - R        :   measurement variance [var_x var_y var_z] (m2)
%   - a        :   acceleration variance (Rule of thumb, between 1/2 - 1 of the max acc between time period intervals)
%   - dt       :   time interval between measurements (seconds)
%   OUTPUT:
%   - coordKF  :   Kalman Filter position coordinates [coord_x coord_y coord_z] (m)
%   - coordRTS :   RTS Smooth position coordinates [coord_x coord_y coord_z] (m)

% Suggested values for bikes
%R = [0.1225 0.1225 0.1225];  % tracking measurement error, std. = 0.35 m
%a = 0.5;                     % acceleration variance (ref: https://doi.org/10.1016/j.trf.2016.04.009)

% Input
n = size(c, 1);                                                                  % number of measurements to consider for smoothing
F = [1 dt 0.5*dt^2 0 0 0 0 0 0; 0 1 dt 0 0 0 0 0 0; 0 0 1 0 0 0 0 0 0; ...       % state transition matrix (motion equations for x and y) 
    0 0 0 1 dt 0.5*dt^2 0 0 0; 0 0 0 0 1 dt 0 0 0; 0 0 0 0 0 1 0 0 0; ...
    0 0 0 0 0 0 1 dt 0.5*dt^2; 0 0 0 0 0 0 0 1 dt; 0 0 0 0 0 0 0 0 1];                     
%Q = [dt^4/4 dt^3/2 dt^2/2 0 0 0 0 0 0; dt^3/2 dt^2 dt 0 0 0 0 0 0; dt^2/2 dt 1 0 0 0 0 0 0; ...   % process noise matrix 
%    0 0 0 dt^4/4 dt^3/2 dt^2/2 0 0 0; 0 0 0 dt^3/2 dt^2 dt 0 0 0; 0 0 0 dt^2 dt 1 0 0 0; ...
%    0 0 0 0 0 0 dt^4/4 dt^3/2 dt^2/2; 0 0 0 0 0 0 dt^3/2 dt^2 dt; 0 0 0 0 0 0 dt^2 dt 1].*a;             
Q = [0.1 0 0 0 0 0 0 0 0; 0 0.1 0 0 0 0 0 0 0; 0 0 1 0 0 0 0 0 0; ...            % process noise matrix (suggested by Luo & Ma)
    0 0 0 0.1 0 0 0 0 0; 0 0 0 0 0.1 0 0 0 0; 0 0 0 0 0 1 0 0 0;
    0 0 0 0 0 0 0.1 0 0; 0 0 0 0 0 0 0 0.1 0; 0 0 0 0 0 0 0 0 1].*a;
c = c';                                                                          % measurement matrix
R = diag(R);                                                                     % mearurement covariance matrix
H = [1 0 0 0 0 0 0 0 0; 0 0 0 1 0 0 0 0 0; 0 0 0 0 0 0 1 0 0];                   % measurement function
X_kf = zeros(size(F, 1), n);                                                     % kf vector
P_kf = zeros(size(F, 1), size(F, 2), n);                                         % kf covariance matrix

% Initialization    
% 1. Initializes the state of the filter
% 2. Initializes belief in the state
s_ini = abs(c(:, 2) - c(:, 1))/dt;
X_kf([1 2 4 5 7 8], 1) = [c(1) s_ini(1) c(2) s_ini(2) c(3) s_ini(3)];
P_kf(:, :, 1) = diag([500, 180, 9, 500, 180, 9, 500, 180, 9]);

for k = 2:n   
 
    % Predict
    % 1. Uses system behavior to predict state at the next time step
    % 2. Adjusts belief to account for the uncertainty in prediction
    X_kf(:, k) = F*X_kf(:, k - 1);
    P_kf(:, :, k) = F*P_kf(:, :, k - 1)*F' + Q;  
    
    % Update
    % 1. Gets a measurement and associated belief about its accuracy (z)
    % 2. Computes residual between estimated state and measurement
    % 3. Computes scaling factor based on whether the measurement or prediction is more accurate (Kalman Gain)
    % 4. Sets state between the prediction and measurement based on scaling factor
    % 5. Updates belief in the state based on how certain we are in the measurement
    y = c(:, k) - H*X_kf(:, k);                         % residual
    K = P_kf(:, :, k)*H'*(H*P_kf(:, :, k)*H' + R)^-1;   % kalman gain
    X_kf(:, k) = X_kf(:, k) + K*y;
    P_kf(:, :, k) = P_kf(:, :, k) - K*H*P_kf(:, :, k);
    
end

% Rauch-Tung-Striebel Smoother
X_rts = zeros(size(F, 1), n);                                               % rts vector
P_rts = zeros(size(F, 1), size(F, 2), n);                                   % rts covariance matrix
X_rts(:, n) = X_kf(:, n);
P_rts(:, :, n) = P_kf(:, :, n);
for k = n-1:-1:1   
 
    % Predict
    Pp = F*P_kf(:, :, k)*F' + Q;
    
    % Update
    C = P_kf(:, :, k)*F'*Pp^-1;
    X_rts(:, k) = X_kf(:, k) + C*(X_rts(:, k + 1) - F*X_kf(:, k));                    
    P_rts(:, :, k) = P_kf(:, :, k) + C*(P_rts(:, :, k + 1) - Pp)*C';  
    
end

% Output
coordKF = X_kf([1 4 7], :)';
coordRTS = X_rts([1 4 7], :)';
 
end
