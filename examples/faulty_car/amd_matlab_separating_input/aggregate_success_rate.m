% aggregate_success_rate.m
%
% Scan results/result_*.mat (written by run_one_config_pace.m, one per
% Slurm array task) and report the success rate over the 84-config sweep.
% A config with no result file is treated as a timeout/non-success -- it
% was killed by the per-task timeout (internal `timeout` or Slurm's own
% --time limit) before Exact_get_u_respon.m returned.
%
% Also writes results_summary.csv (config_idx, width, alpha_hi, cx, cy,
% success, problem_code, t_elapsed) mirroring the column layout used by
% pace_runtime_scaling_package's admire_success_rate_summary*.csv.

here = fileparts(mfilename('fullpath'));
S = load(fullfile(here, 'configs.mat'));
configs = S.configs;
n_total = size(configs, 1);

rows = cell(n_total, 1);
n_success = 0;
n_missing = 0;
n_ran_but_failed = 0;

for idx = 1:n_total
    f = fullfile(here, 'results', sprintf('result_%03d.mat', idx));
    if ~isfile(f)
        n_missing = n_missing + 1;
        rows{idx} = sprintf('%d,%.4f,%.4f,%.4f,%.4f,0,NA,timeout_or_missing,NA', ...
            idx, configs(idx,1), configs(idx,2), configs(idx,3), configs(idx,4));
        continue
    end
    R = load(f); r = R.result;
    if r.success
        n_success = n_success + 1;
    else
        n_ran_but_failed = n_ran_but_failed + 1;
    end
    rows{idx} = sprintf('%d,%.4f,%.4f,%.4f,%.4f,%d,%d,%s,%.3f', ...
        idx, r.width, r.alpha_hi, r.cx, r.cy, r.success, r.problem_code, ...
        strrep(r.info, ',', ';'), r.t_elapsed);
end

fid = fopen(fullfile(here, 'results_summary.csv'), 'w');
fprintf(fid, 'config_idx,width,alpha_hi,cx,cy,success,problem_code,info,t_elapsed_s\n');
for idx = 1:n_total
    fprintf(fid, '%s\n', rows{idx});
end
fclose(fid);

n_done = n_total - n_missing;
fprintf('\n==================== Faulty-car AMD sweep summary ====================\n');
fprintf('Total configs        : %d\n', n_total);
fprintf('Completed (has result): %d\n', n_done);
fprintf('  - succeeded (sol.problem==0): %d\n', n_success);
fprintf('  - ran, not separable/other  : %d\n', n_ran_but_failed);
fprintf('Timed out / no result file    : %d\n', n_missing);
fprintf('------------------------------------------------------------------------\n');
fprintf('Success rate (of all %d configs): %.1f%%  (%d/%d)\n', ...
    n_total, 100*n_success/n_total, n_success, n_total);
if n_done > 0
    fprintf('Success rate (of %d completed) : %.1f%%  (%d/%d)\n', ...
        n_done, 100*n_success/n_done, n_success, n_done);
end
fprintf('========================================================================\n');
fprintf('Wrote results_summary.csv\n');
