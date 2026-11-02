function convert_bu100_matlab(output)
% Preserve raw records and convert full ZNE traces using the frozen FIR.
if nargin == 0
    output = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'outputs', 'cspp_bu100_20260910');
end
p = jsondecode(fileread(fullfile(output, 'protocol.json')));
target = fullfile(output, 'waveforms_100hz_zne.h5');
prior = [];
if isfile(target)
    statuspath = fullfile(output, 'conversion_status.json');
    assert(isfile(statuspath), 'Existing HDF5 has no completed conversion status');
    prior = jsondecode(fileread(statuspath));
    assert(numel(prior) == numel(p.records) && isequal({prior.id}, {p.records.id}), 'Prior conversion cohort differs');
end
results = cell(numel(p.records), 1);
for i = 1:numel(p.records)
    r = p.records(i);
    s = struct('id', r.id, 'cohort', r.cohort, 'instrument', r.instrument, ...
        'record_key', r.record_key, 'reference_p_seconds', r.reference_p_seconds, ...
        'status', 'excluded', 'reason', '', 'source_sample_rate_hz', 0, ...
        'source_samples', 0, 'target_samples', 0, 'target_sample_rate_hz', 100, ...
        'source_start_text', '', 'source_units', '', 'fir_group_delay_seconds', 0, ...
        'absolute_start_provided', false, 'component_alignment_basis', '', ...
        'source_padding_values_removed', 0);
    if ~isempty(prior) && strcmp(prior(i).status, 'converted')
        names = fieldnames(prior(i));
        for k = 1:numel(names)
            s.(names{k}) = prior(i).(names{k});
        end
        info = h5info(target, ['/' r.id '/wave']);
        assert(isequal(info.Dataspace.Size, [s.target_samples, 3]), 'Existing dataset dimensions differ');
        results{i} = s;
        continue;
    end
    try
        if strcmp(r.instrument, 'intensity')
            [x, fs, start, units] = read_intensity(r.path_txt);
            s.component_alignment_basis = 'Shared rows in one three-component file';
        else
            paths = {r.path_z, r.path_n, r.path_e};
            comp = {'UD', 'NS', 'EW'};
            parts = cell(1, 3); rates = zeros(1, 3); starts = cell(1, 3); unitlist = cell(1, 3);
            identities = cell(1, 3); padding = zeros(1, 3);
            for c = 1:3
                [parts{c}, rates(c), starts{c}, unitlist{c}, identities{c}, padding(c)] = read_strong(paths{c}, comp{c});
            end
            assert(all(rates == rates(1)), 'Component sample rates differ');
            assert(all(cellfun(@numel, parts) == numel(parts{1})), 'Component sample counts differ');
            assert(strcmp(starts{1}, starts{2}) && strcmp(starts{1}, starts{3}), 'Component start times differ');
            assert(strcmp(unitlist{1}, unitlist{2}) && strcmp(unitlist{1}, unitlist{3}), 'Component units differ');
            assert(strcmp(identities{1}, identities{2}) && strcmp(identities{1}, identities{3}), 'Component event or pre-event metadata differ');
            x = [parts{1}, parts{2}, parts{3}]; fs = rates(1); start = starts{1}; units = unitlist{1};
            s.source_padding_values_removed = sum(padding);
            s.component_alignment_basis = 'Matched record ID, event header, pre-event field, sample rate and count';
            if ~isempty(start)
                s.component_alignment_basis = [s.component_alignment_basis ', and recorder time'];
            end
        end
        assert(all(isfinite(x), 'all'), 'Nonfinite source values');
        assert(size(x, 2) == 3 && size(x, 1) >= fs, 'Invalid waveform shape or length');
        assert(isfinite(r.reference_p_seconds) && r.reference_p_seconds >= 0 && ...
            r.reference_p_seconds < size(x, 1) / fs, 'Reference outside record');
        s.source_sample_rate_hz = fs; s.source_samples = size(x, 1);
        s.source_start_text = start; s.source_units = units;
        s.absolute_start_provided = ~isempty(start);
        if abs(fs - 200) < 1e-7
            y = filter(p.fir_coefficients(:), 1, x, [], 1);
            y = y(1:2:end, :);
            y(1:15, :) = NaN;
            s.fir_group_delay_seconds = p.design.fir_group_delay_seconds;
        elseif abs(fs - 100) < 1e-7
            y = x;
        elseif abs(fs - 500) < 1e-7 && isfield(p, 'resampling_500hz')
            spec = p.resampling_500hz;
            assert(spec.source_sample_rate_hz == 500 && spec.target_sample_rate_hz == 100);
            y = filter(spec.coefficients(:), 1, x, [], 1);
            y = y(1:5:end, :);
            y(1:spec.initial_target_samples_withheld, :) = NaN;
            s.fir_group_delay_seconds = spec.group_delay_seconds;
        else
            error('Unsupported sample rate %.9g', fs);
        end
        y = single(y);
        dataset = ['/' r.id '/wave'];
        % MATLAB Nx3 is exposed as 3xN by h5py; verified independently.
        h5create(target, dataset, size(y), 'Datatype', 'single', ...
            'ChunkSize', [min(size(y, 1), 10000), 3], 'Deflate', 4);
        h5write(target, dataset, y);
        h5writeatt(target, dataset, 'component_order', 'ZNE');
        h5writeatt(target, dataset, 'sample_rate_hz', 100);
        h5writeatt(target, dataset, 'reference_p_seconds', r.reference_p_seconds);
        h5writeatt(target, dataset, 'source_units', units);
        s.target_samples = size(y, 1); s.status = 'converted';
    catch exception
        s.reason = exception.message;
        fprintf('EXCLUDED %s: %s\n', r.id, s.reason);
    end
    results{i} = s;
    if mod(i, 50) == 0 || i == numel(p.records)
        fprintf('Converted/scanned %d/%d\n', i, numel(p.records));
    end
end
status = vertcat(results{:});
f = fopen(fullfile(output, 'conversion_status.json'), 'w', 'n', 'UTF-8');
assert(f >= 0); cleanup = onCleanup(@() fclose(f));
fprintf(f, '%s\n', jsonencode(status, PrettyPrint=true));
clear cleanup;
writetable(struct2table(status), fullfile(output, 'conversion_status.csv'));
end

function [x, fs, start, units, identity, padding] = read_strong(path, component)
f = fopen(path, 'r'); assert(f >= 0, 'Cannot open source'); cleanup = onCleanup(@() fclose(f));
head = cell(16, 1);
for j = 1:16
    head{j} = fgetl(f); assert(ischar(head{j}), 'Truncated header');
end
header = strjoin(head, newline);
c = regexp(header, 'COMP\.\s*(\S+)', 'tokens', 'once');
assert(~isempty(c) && strcmp(c{1}, component), 'Component header mismatch');
n = regexp(header, 'NO\. OF POINTS:\s*(\d+)\s+EQUALLY SPACED INTERVALS OF:\s*([\d.Ee+-]+)', 'tokens', 'once');
assert(~isempty(n), 'Missing sample metadata');
fs = 1 / str2double(n{2});
t = regexp(header, 'RECORDER TIME:\s*([^\r\n]+)', 'tokens', 'once');
start = '';
if ~isempty(t)
    start = strtrim(t{1});
end
pre = regexp(header, 'PRE-EVENT TIME:\s*(\S+)\s*SEC', 'tokens', 'once');
assert(~isempty(pre), 'Missing pre-event field');
assert(~isempty(start) || isfinite(str2double(pre{1})), 'No recorder time or finite pre-event duration');
identity = [strtrim(head{2}) '|' pre{1}];
u = regexp(header, 'UNIT:\s*([^\r\n]+)', 'tokens', 'once');
assert(~isempty(u), 'Missing unit'); units = strtrim(u{1});
x = fscanf(f, '%f');
declared = str2double(n{1}); padding = numel(x) - declared;
if padding ~= 0
    assert(padding > 0 && padding < 8 && numel(x) == ceil(declared / 8) * 8 && ...
        all(x(declared + 1:end) == 0), 'Unexplained header and numeric sample count mismatch');
    x = x(1:declared);
end
end

function [x, fs, start, units] = read_intensity(path)
f = fopen(path, 'r'); assert(f >= 0, 'Cannot open source'); cleanup = onCleanup(@() fclose(f));
head = cell(6, 1);
for j = 1:6
    head{j} = fgetl(f); assert(ischar(head{j}) && startsWith(head{j}, '#'), 'Unexpected intensity header');
end
header = strjoin(head, newline);
assert(~isempty(regexp(header, 'DataZ\s+DataE\s+DataN', 'once')), 'Unknown column order');
a = regexp(header, 'sample\s*rate:\s*([\d.]+)', 'tokens', 'once');
t = regexp(header, 'start\s*time:\s*([^\r\n]+)', 'tokens', 'once');
assert(~isempty(a) && ~isempty(t), 'Missing intensity sampling or time metadata');
fs = str2double(a{1}); start = strtrim(t{1}); units = 'unverified_source_units';
v = textscan(f, '%f%f%f', 'Delimiter', ',', 'CollectOutput', true, 'ReturnOnError', false);
x = v{1}(:, [1 3 2]);
end
