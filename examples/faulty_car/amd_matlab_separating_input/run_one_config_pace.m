function run_one_config_pace(idx)
% run_one_config_pace  Solve the AMD separating-input MILP (exact,
% unmodified Exact_get_u_respon.m / StateSpace.m -- see amd_lib/, byte-
% identical to AMD-code's originals) for sweep config `idx` and save the
% result to results/result_<idx>.mat.
%
% Meant to be run as one Slurm array task per config (see
% run_faulty_car_amd_sweep.sbatch), so each config gets its own full
% wall-clock budget instead of sharing one job's clock. YALMIP and Gurobi
% locations come from env vars set by the sbatch script (YALMIP_DIR,
% GUROBI_MATLAB_DIR) rather than being hardcoded, since those paths differ
% between this workstation and PACE.

here = fileparts(mfilename('fullpath'));

yalmip_dir = getenv('YALMIP_DIR');
gurobi_matlab_dir = getenv('GUROBI_MATLAB_DIR');
if isempty(yalmip_dir) || isempty(gurobi_matlab_dir)
    error(['YALMIP_DIR and GUROBI_MATLAB_DIR env vars must be set before ' ...
           'calling run_one_config_pace (see run_faulty_car_amd_sweep.sbatch).']);
end

addpath(genpath(yalmip_dir));
addpath(gurobi_matlab_dir);
addpath(here);
addpath(fullfile(here, 'amd_lib'));

resultsdir = fullfile(here, 'results');
if ~exist(resultsdir, 'dir')
    mkdir(resultsdir);
end

S = load(fullfile(here, 'configs.mat'));
configs = S.configs;

if ischar(idx) || isstring(idx)
    idx = str2double(idx);
end

cfg = configs(idx, :);
width = cfg(1); alpha_hi = cfg(2); cx = cfg(3); cy = cfg(4);

fprintf('Config %d: width=%.4f alpha_hi=%.4f cx=%.4f cy=%.4f\n', ...
    idx, width, alpha_hi, cx, cy);

[modes, bounds, T_hor, epsi, NNorm] = faulty_car_config(width, alpha_hi, cx, cy);

t_start = tic;
try
    [u_star, sol] = Exact_get_u_respon(modes, bounds, T_hor, epsi, NNorm);
    t_elapsed = toc(t_start);
    success = (sol.problem == 0);
    problem_code = sol.problem;
    info = sol.info;
    errmsg = '';
catch ME
    t_elapsed = toc(t_start);
    success = false;
    problem_code = -999;
    info = 'MATLAB error';
    errmsg = ME.message;
    u_star = [];
end

result = struct('config_idx', idx, 'width', width, 'alpha_hi', alpha_hi, ...
    'cx', cx, 'cy', cy, 'success', success, 'problem_code', problem_code, ...
    'info', info, 'errmsg', errmsg, 't_elapsed', t_elapsed, 'u_star', u_star);

save(fullfile(resultsdir, sprintf('result_%03d.mat', idx)), 'result');

fprintf('Config %d DONE: success=%d problem=%d info=%s t=%.2fs\n', ...
    idx, success, problem_code, info, t_elapsed);

end
