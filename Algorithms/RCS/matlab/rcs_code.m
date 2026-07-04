clear
clc

I_rocket = 0.95625;
T = 0.1;

% Continuous dynamics: state = [theta; omega]
A = [0 1; 0 0];
B = [0; 1/I_rocket];

% Discretize
sys_c = ss(A, B, eye(2), 0);
sys_d = c2d(sys_c, T, 'zoh');
Ad = sys_d.A;
Bd = sys_d.B;

max_torque= 1.8;
max_acceptable_omega= 0.16; % roll rate
max_acceptable_theta=0.02; %roll angle

q_theta = 1 / (max_acceptable_theta)^2;
q_omega = 1 / (max_acceptable_omega)^2;
%R       = 1 / (max_torque)^2;           


% Cost matrices
Q = diag([5*q_theta, 2*q_omega]);
R = 1 / (max_torque)^2;

% Solve DARE
[K, ~, ~] = dlqr(Ad, Bd, Q, R);

% K = [k_theta, k_omega]
disp(K)

% Simulation parameters
dt = 0.001;           % integration step
T_ctrl = 0.1;         % control decision every 0.1s
t_end = 30.0;
N = t_end / dt;

% Initial conditions
theta = 0.05;         
omega = -0.1;

% Deadband thresholds 

dead_theta = 0.025;    % rad
dead_omega = 0.08;     % rad/s


fire_threshold = 0.8; % switching variable threshold

log_t = zeros(1,N); log_theta = zeros(1,N);
log_omega = zeros(1,N); log_u = zeros(1,N);

u_current = 0;        % torque currently applied
ctrl_timer = 0;



for k = 1:N
    t = k * dt;
    ctrl_timer = ctrl_timer + dt;

    % Control decision every T_ctrl seconds
    if ctrl_timer >= T_ctrl
        ctrl_timer = 0;

        % Add simulated sensor noise
        theta_m = theta + 0.0087 * randn();
        omega_m = omega + 0.05  * randn();

        % Deadband check
        if abs(theta_m) < dead_theta && abs(omega_m) < dead_omega
            u_current = 0;
        else
            % LQR switching variable
            sv = -K(1)*theta_m - K(2)*omega_m;

            if sv > fire_threshold
                u_current = 1.8;
            elseif sv < -fire_threshold
                u_current = -1.8;
            else
                u_current = 0;
            end
        end
    end

    % Integrate dynamics
    alpha_now = u_current / I_rocket;
    omega = omega + alpha_now * dt;
    theta = theta + omega * dt;

    log_t(k) = t;
    log_theta(k) = theta;
    log_omega(k) = omega;
    log_u(k) = u_current;
end

% Plot
figure;
subplot(3,1,1); plot(log_t, rad2deg(log_theta)); ylabel('Angle (deg)'); grid on;
subplot(3,1,2); plot(log_t, rad2deg(log_omega)); ylabel('Rate (rad/s)'); grid on;
subplot(3,1,3); plot(log_t, log_u); ylabel('Torque (Nm)'); grid on;
xlabel('Time (s)');