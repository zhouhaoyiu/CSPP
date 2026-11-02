function summarize_bu100_matlab(output)
if nargin == 0
    output = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'outputs', 'cspp_bu100_20260910');
end
t = readtable(fullfile(output, 'record_metrics.csv'), 'TextType', 'string');
groups = ["all"; unique(t.instrument); unique(t.cohort)];
rows = cell(0, 1);
for g = groups'
    if g == "all"
        mask = true(height(t), 1);
    else
        mask = t.instrument == g | t.cohort == g;
    end
    for stratum = ["all_records", "reference_ge_10s", "gate_allowed_at_reference"]
        selected = mask;
        if stratum == "reference_ge_10s"
            selected = selected & t.reference_p_seconds >= 10;
        elseif stratum == "gate_allowed_at_reference"
            selected = selected & ismember(lower(string(t.candidate_allowed_at_reference)), ["true", "1"]);
        end
        q = t(selected, :);
        for tol = ["1500ms", "500ms", "250ms"]
            hits = q.("matched_" + tol) == 1;
            errors = numeric_values(q.("error_" + tol)(hits));
            latencies = numeric_values(q.("latency_" + tol)(hits));
            row = struct('group', g, 'stratum', stratum, 'tolerance', tol, ...
                'records', height(q), 'matched', sum(hits), ...
                'recall', sum(hits) / height(q), 'candidate_count', sum(q.candidate_count), ...
                'unmatched_candidates', sum(q.candidate_count) - sum(hits), ...
                'median_absolute_error_seconds', median(abs(errors)), ...
                'mean_absolute_error_seconds', mean(abs(errors)), ...
                'median_signed_error_seconds', median(errors), ...
                'median_availability_latency_seconds', median(latencies), ...
                'evaluated_hours', sum(q.evaluated_seconds) / 3600);
            rows{end + 1, 1} = row;
        end
    end
end

function values = numeric_values(values)
if ~isnumeric(values)
    values = str2double(string(values));
end
end
summary = struct2table(vertcat(rows{:}));
writetable(summary, fullfile(output, 'summary.csv'));
disp(summary(summary.tolerance == "1500ms", :));
end
