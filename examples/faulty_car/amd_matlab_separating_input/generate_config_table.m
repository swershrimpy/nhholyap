% generate_config_table.m
%
% Build the 84-config sweep table for the faulty-car AMD separating-input
% study: 3 initial-interval widths x 4 actuator-fault authority levels x
% 7 initial-position centres = 84 configs.
%
% Saves faulty_car_sweep/configs.mat (variable `configs`, 84x4: [width,
% alpha_hi, cx, cy]) and configs.csv for readability.

widths    = [0.05, 0.10, 0.20];             % initial-box half-width (m), px/py/phi
alpha_his = [0.1, 0.3, 0.5, 0.7];           % Actuator Fault alpha in [0, alpha_hi]
radius    = 0.5;                            % ring radius for off-origin centres (m)
angles    = (0:5) * (pi/3);                 % 6 points at 60-degree spacing
centers   = [0, 0; radius*cos(angles(:)), radius*sin(angles(:))];  % 7 x 2 [cx, cy]

configs = zeros(numel(widths)*numel(alpha_his)*size(centers,1), 4);
idx = 0;
for iw = 1:numel(widths)
    for ia = 1:numel(alpha_his)
        for ic = 1:size(centers,1)
            idx = idx + 1;
            configs(idx, :) = [widths(iw), alpha_his(ia), centers(ic,1), centers(ic,2)];
        end
    end
end

assert(idx == 84, 'Expected 84 configs, got %d', idx);

save(fullfile(fileparts(mfilename('fullpath')), 'configs.mat'), 'configs');

fid = fopen(fullfile(fileparts(mfilename('fullpath')), 'configs.csv'), 'w');
fprintf(fid, 'config_idx,width,alpha_hi,cx,cy\n');
for k = 1:size(configs,1)
    fprintf(fid, '%d,%.4f,%.4f,%.4f,%.4f\n', k, configs(k,1), configs(k,2), configs(k,3), configs(k,4));
end
fclose(fid);

fprintf('Wrote %d configs to configs.mat / configs.csv\n', size(configs,1));
