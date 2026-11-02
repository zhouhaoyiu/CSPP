function scan_bu100_matlab(output)
if nargin == 0
    output = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'outputs', 'cspp_bu100_threshold_scan_20260911');
end
p = jsondecode(fileread(fullfile(output, 'protocol.json')));
qs = [p.threshold_pairs.candidate_threshold]; es = [p.threshold_pairs.event_threshold];
tolerances = p.tolerances_seconds(:)';
selftest(output, qs, es, tolerances);
metadata = readtable(fullfile(p.base_directory, 'record_metrics.csv'), 'TextType', 'string');
n = height(metadata); k = numel(qs); nt = numel(tolerances);
counts = zeros(n, k); matches = false(n, k, nt);
errors = NaN(n, k, nt); latencies = errors;
cache = fullfile(output, 'packet_scores.h5');
baseline = find(qs == 0.1 & es == 0.975); assert(isscalar(baseline));
for i = 1:n
    scores = h5read(cache, '/' + metadata.id(i) + '/scores');
    [counts(i, :), m, e, d] = bu100_scan_record(scores, metadata.reference_p_seconds(i), qs, es, tolerances);
    matches(i, :, :) = reshape(m, 1, k, nt);
    errors(i, :, :) = reshape(e, 1, k, nt);
    latencies(i, :, :) = reshape(d, 1, k, nt);
    assert(counts(i, baseline) == metadata.candidate_count(i));
    suffixes = ["1500ms", "500ms", "250ms"];
    for z = 1:nt
        assert(m(baseline, z) == metadata.("matched_" + suffixes(z))(i));
        original = metadata.("error_" + suffixes(z))(i);
        assert(isequaln(e(baseline, z), original) || abs(e(baseline, z) - original) < 1e-12);
    end
    if mod(i, 250) == 0 || i == n
        fprintf('Threshold scan %d/%d records, %d pairs\n', i, n, k);
    end
end
target = fullfile(output, 'scan_record_metrics.h5');
assert(~isfile(target), 'Refusing to overwrite scan arrays');
h5create(target, '/candidate_count', size(counts), 'Datatype', 'uint16'); h5write(target, '/candidate_count', uint16(counts));
h5create(target, '/matched', size(matches), 'Datatype', 'uint8'); h5write(target, '/matched', uint8(matches));
h5create(target, '/error_seconds', size(errors)); h5write(target, '/error_seconds', errors);
h5create(target, '/latency_seconds', size(latencies)); h5write(target, '/latency_seconds', latencies);
writetable(metadata(:, {'id', 'record_key', 'cohort', 'instrument', 'reference_p_seconds'}), fullfile(output, 'record_order.csv'));
groups = ["all"; unique(metadata.instrument); unique(metadata.cohort)];
strata = ["all_records", "reference_ge_10s"];
rows = cell(numel(groups) * numel(strata) * nt, 1); at = 0;
for g = groups'
    for stratum = strata
        selected = g == "all" | metadata.instrument == g | metadata.cohort == g;
        if stratum == "reference_ge_10s"
            selected = selected & metadata.reference_p_seconds >= 10;
        end
        total = sum(selected); candidate_count = sum(counts(selected, :), 1)';
        for z = 1:nt
            tp = sum(matches(selected, :, z), 1)'; fp = candidate_count - tp; fn = total - tp;
            precision = tp ./ candidate_count; recall = tp / total; f1 = 2 * tp ./ (total + candidate_count);
            mae = mean(abs(errors(selected, :, z)), 1, 'omitnan')';
            median_error = median(abs(errors(selected, :, z)), 1, 'omitnan')';
            median_latency = median(latencies(selected, :, z), 1, 'omitnan')';
            row = table(repmat(g, k, 1), repmat(stratum, k, 1), repmat(tolerances(z), k, 1), ...
                qs', es', repmat(total, k, 1), tp, fp, fn, candidate_count, precision, recall, f1, mae, median_error, median_latency, ...
                'VariableNames', {'group', 'stratum', 'tolerance_seconds', 'candidate_threshold', 'event_threshold', ...
                'records', 'TP', 'FP_proxy', 'FN', 'candidate_count', 'precision', 'recall', 'F1', ...
                'mean_absolute_error_seconds', 'median_absolute_error_seconds', 'median_availability_latency_seconds'});
            at = at + 1; rows{at} = row;
        end
    end
end
summary = vertcat(rows{:});
writetable(summary, fullfile(output, 'threshold_summary.csv'));
all_primary = summary(summary.group == "all" & summary.stratum == "all_records" & summary.tolerance_seconds == 1.5, :);
ranking = sortrows(all_primary, {'F1', 'recall', 'event_threshold', 'candidate_threshold'}, {'descend', 'descend', 'descend', 'descend'});
fixed = all_primary(all_primary.candidate_threshold == 0.1 & all_primary.event_threshold == 0.975, :);
one_dimensional = ranking(ranking.candidate_threshold == 0.1, :);
selected = [fixed; one_dimensional(1, :); ranking(1, :)];
selected = addvars(selected, ["frozen"; "best_event_only"; "best_joint"], 'Before', 1, 'NewVariableNames', 'selection');
writetable(selected, fullfile(output, 'selected_operating_points.csv'));
writetable(ranking, fullfile(output, 'ranked_all_records_1500ms.csv'));
disp(selected);
end

function selftest(output, qs, es, tolerances)
expected = readtable(fullfile(output, 'scan_selftest_expected.csv'), 'TextType', 'string');
for id = unique(expected.id)'
    cache = fullfile(output, 'scan_selftest.h5');
    scores = h5read(cache, '/' + id + '/scores');
    ref = h5readatt(cache, '/' + id, 'reference_p_seconds');
    [count, m, e, d] = bu100_scan_record(scores, ref, qs, es, tolerances);
    subset = expected(expected.id == id, :);
    for j = 1:height(subset)
        a = find(qs == subset.candidate_threshold(j) & es == subset.event_threshold(j));
        b = find(tolerances == subset.tolerance(j));
        assert(count(a) == subset.candidate_count(j) && m(a, b) == subset.matched(j));
        assert(isequaln(e(a, b), subset.error(j)) || abs(e(a, b) - subset.error(j)) < 1e-12);
        assert(isequaln(d(a, b), subset.latency(j)) || abs(d(a, b) - subset.latency(j)) < 1e-12);
    end
end
fprintf('MATLAB sweep matches scalar PacketCandidateState for %d synthetic comparisons\n', height(expected));
end
