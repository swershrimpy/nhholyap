function [modes, bounds, T_hor, epsi, NNorm] = faulty_car_config(width, alpha_hi, cx, cy)
% faulty_car_config  Build the (modes, bounds) AMD problem data for the
% faulty nonholonomic car, parametrised by:
%
%   width     - half-width of the initial-state box on px, py, and phi
%               (box is centred at [cx, cy, 0])
%   alpha_hi  - upper bound of the Actuator Fault steering-authority
%               interval alpha in [0, alpha_hi]  (alpha=1 => nominal)
%   cx, cy    - centre of the initial px, py box
%
% This is model_faulty_car.m (sections 1-6) turned into a function so it
% can be swept over many configurations. The AMD algorithm/optimizer
% (StateSpace.m, Exact_get_u_respon.m) is NOT modified; only the problem
% data (initial-state box and Actuator-Fault authority) varies between
% configs. All other parameters (dt, horizon, epsilon, norm, velocity/
% noise bounds, mode structure) are identical to model_faulty_car.m.

%% Fixed problem parameters (identical to model_faulty_car.m)
dt      = 0.5;    % sampling time (s)
T_hor   = 10;     % horizon: 10 steps
epsi    = 0.05;   % separability threshold (m)
NNorm   = 1;      % objective norm (L1)

v0  = 0.5;        % forward velocity at linearisation point (m/s)

v_max     = 1.0;
omega_max = 1.0;

w_pos     = 0.001;   % positional process-noise bound (m/step)
w_phi_nom = 0.001;   % heading noise (rad/step), Nominal & Sensor Fault

v_meas = 0.01;       % measurement-noise bound (m)

%% Swept initial-state box: centre [cx, cy, 0], half-width `width`
px_lo = cx - width;   px_hi = cx + width;
py_lo = cy - width;   py_hi = cy + width;
ph_lo = -width;        ph_hi = width;

%% Swept Actuator Fault authority: alpha in [0, alpha_hi]
alpha_c   = alpha_hi / 2;                 % centre of the alpha interval
w_phi_act = alpha_c * omega_max * dt;     % residual heading-noise bound

%% Linearised discrete-time matrices (identical structure to model_faulty_car.m)
Ad = [1,  0,      0;
      0,  1,  v0*dt;
      0,  0,      1];

Bu_nom  = dt * [1,    0;
                0,    0;
                0,  1.0];    % Nominal / Sensor Fault: alpha = 1.0
Bu_act  = dt * [1,    0;
                0,    0;
                0, alpha_c]; % Actuator Fault: alpha = alpha_c (centre)

Bd_dum = zeros(3, 1);
Bw     = dt * eye(3);
Bv     = zeros(3, 2);

B_nom  = [Bu_nom,  Bd_dum, Bw, Bv];
B_act  = [Bu_act,  Bd_dum, Bw, Bv];
B_sens = B_nom;

C_nom  = [1, 0, 0;
          0, 1, 0];
C_sens = 0.95 * C_nom;

D_all  = [0, 0,  0,  0, 0, 0,  1, 0;
          0, 0,  0,  0, 0, 0,  0, 1];

g_nom  = [0;   0  ];
g_act  = [0;   0  ];
g_sens = [0.2; 0.2];

f_all  = zeros(3, 1);

%% StateSpace objects
sys_nom  = StateSpace(Ad, B_nom,  C_nom,  D_all, f_all, g_nom);
sys_act  = StateSpace(Ad, B_act,  C_nom,  D_all, f_all, g_act);
sys_sens = StateSpace(Ad, B_sens, C_sens, D_all, f_all, g_sens);

%% Global bounds
P_0 = [ 1, 0, 0;
       -1, 0, 0;
        0, 1, 0;
        0,-1, 0;
        0, 0, 1;
        0, 0,-1];
p_0 = [px_hi; -px_lo; py_hi; -py_lo; ph_hi; -ph_lo];

P_x = [ 1, 0, 0;
       -1, 0, 0;
        0, 1, 0;
        0,-1, 0;
        0, 0, 1;
        0, 0,-1];
p_x = [10; 10; 10; 10; pi; pi];

Q_u = [ 1, 0;
       -1, 0;
        0, 1;
        0,-1];
q_u = [v_max; v_max; omega_max; omega_max];

Q_w = [ 1, 0, 0;
       -1, 0, 0;
        0, 1, 0;
        0,-1, 0;
        0, 0, 1;
        0, 0,-1];

Q_v = [ 1, 0;
       -1, 0;
        0, 1;
        0,-1];
q_v = repmat(v_meas, 4, 1);

bounds = struct('P_0', P_0, 'p_0', p_0, ...
                'P_x', P_x, 'p_x', p_x, ...
                'Q_u', Q_u, 'q_u', q_u, ...
                'Q_w', Q_w, 'q_w', repmat(w_pos, 6, 1), ...
                'Q_v', Q_v, 'q_v', q_v);

%% Mode-specific bounds
P_y = zeros(0, 0);
p_y = zeros(0, 1);

Q_d = [1; -1];
q_d = [0; 0];

q_w_nom = [w_pos; w_pos; w_pos; w_pos; w_phi_nom; w_phi_nom];
q_w_act = [w_pos; w_pos; w_pos; w_pos; w_phi_act; w_phi_act];

bounds_nom  = struct('P_y', P_y, 'p_y', p_y, ...
                     'Q_d', Q_d, 'q_d', q_d, ...
                     'Q_w', Q_w, 'q_w', q_w_nom, ...
                     'Q_v', Q_v, 'q_v', q_v);

bounds_act  = struct('P_y', P_y, 'p_y', p_y, ...
                     'Q_d', Q_d, 'q_d', q_d, ...
                     'Q_w', Q_w, 'q_w', q_w_act, ...
                     'Q_v', Q_v, 'q_v', q_v);

bounds_sens = struct('P_y', P_y, 'p_y', p_y, ...
                     'Q_d', Q_d, 'q_d', q_d, ...
                     'Q_w', Q_w, 'q_w', q_w_nom, ...
                     'Q_v', Q_v, 'q_v', q_v);

%% Pack modes array
struct_nom  = struct('name', 'Nominal',        'sys', sys_nom,  'bounds', bounds_nom);
struct_act  = struct('name', 'Actuator Fault', 'sys', sys_act,  'bounds', bounds_act);
struct_sens = struct('name', 'Sensor Fault',   'sys', sys_sens, 'bounds', bounds_sens);

modes = [struct_nom, struct_act, struct_sens];

end
