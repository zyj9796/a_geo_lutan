args = argv();
if numel(args) ~= 3
  error('Usage: octave attach_lutan_deformation.m defo_ssa.mat geobc_points.csv output.csv');
endif

mat_path = args{1};
points_path = args{2};
output_path = args{3};

load(mat_path, 'defo');
points = dlmread(points_path, ',', 1, 0);

% defo stores one-based range-column/azimuth-row; Geo-BC tables store
% zero-based azimuth-row/range-column.
lookup = zeros(2000, 1000, 'uint32');
source_linear = sub2ind([2000, 1000], round(defo(:, 2)), round(defo(:, 1)));
lookup(source_linear) = uint32(1:rows(defo));

query_row = round(points(:, 2)) + 1;
query_col = round(points(:, 3)) + 1;
inside = query_row >= 1 & query_row <= 2000 & query_col >= 1 & query_col <= 1000;
source_index = zeros(rows(points), 1, 'uint32');
source_index(inside) = lookup(sub2ind([2000, 1000], query_row(inside), query_col(inside)));
matched = source_index > 0;
idx = double(source_index(matched));

epoch_datenum = datenum({
  '2025-01-04', '2025-01-24', '2025-02-01', '2025-02-21', '2025-03-21', ...
  '2025-04-18', '2025-05-16', '2025-06-13', '2025-07-11', '2025-09-05', ...
  '2025-10-31', '2025-11-28', '2025-12-26', '2026-01-03', '2026-01-23', ...
  '2026-03-28', '2026-04-17', '2026-04-25', '2026-05-15', '2026-05-23'
});
t = (epoch_datenum - epoch_datenum(1)) / 365.25;
tc = t - mean(t);
displacement = defo(idx, 6:25);
velocity = displacement * tc(:) / sum(tc .^ 2);
cumulative = displacement(:, end) - displacement(:, 1);

out = [points(matched, :), defo(idx, 3:5), velocity, cumulative];
fid = fopen(output_path, 'w');
fprintf(fid, ['fid,row,col,method_lon,method_lat,method_height_m,' ...
  'gamma_dsm_lon,gamma_dsm_lat,gamma_dsm_height_m,gamma_dsm_ok,' ...
  'gamma_dsm_residual,' ...
  'lutan_lon,lutan_lat,lutan_height_m,deformation_rate_mm_yr,cumulative_deformation_mm\n']);
fclose(fid);
dlmwrite(output_path, out, '-append', 'delimiter', ',', 'precision', '%.10g');

fprintf('geobc_points=%d matched_deformation=%d unmatched=%d\n', rows(points), sum(matched), sum(!matched));
