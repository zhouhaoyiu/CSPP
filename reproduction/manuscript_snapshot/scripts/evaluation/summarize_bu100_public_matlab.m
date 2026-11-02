function summarize_bu100_public_matlab(root)
% Aggregate the frozen public-model candidate counts on the common cohort.
if nargin == 0
    root = 'workspace/outputs/cspp_bu100_public_baselines_20260911';
end
models = ["phasenet_instance", "eqtransformer_original"];
rows = {};
selected = {};
for model = models
    records = readtable(fullfile(root, model, 'record_metrics.csv'), 'TextType', 'string');
    assert(all(records.tp + records.fn == 1));
    assert(all(records.tp + records.fp_proxy == records.candidate_count));
    groups = ["all"; "strong_motion"; "intensity"; unique(records.cohort)];
    thresholds = unique(records.threshold);
    for threshold = thresholds'
        for tolerance = [1.5, 0.5, 0.25]
            atPoint = records.threshold == threshold & records.tolerance == tolerance;
            assert(nnz(atPoint) == 2136);
            for group = groups'
                take = atPoint & (group == "all" | records.instrument == group | records.cohort == group);
                t = records(take,:);
                s = struct('model',model,'threshold',threshold, ...
                    'is_default',t.is_default(1),'tolerance',tolerance,'group',group, ...
                    'records',height(t),'tp',sum(t.tp),'fp_proxy',sum(t.fp_proxy),'fn',sum(t.fn));
                s.precision = s.tp / (s.tp + s.fp_proxy);
                s.recall = s.tp / (s.tp + s.fn);
                s.f1 = 2*s.tp / (2*s.tp + s.fp_proxy + s.fn);
                s.mean_absolute_error = mean(t.absolute_error,'omitnan');
                s.median_absolute_error = median(t.absolute_error,'omitnan');
                s.p95_absolute_error = prctile(t.absolute_error,95);
                s.median_availability_latency = median(t.availability_latency,'omitnan');
                s.p95_availability_latency = prctile(t.availability_latency,95);
                rows{end+1} = s; %#ok<AGROW>
            end
        end
    end
    tableRows = struct2table([rows{:}]);
    eligible = tableRows(tableRows.model == model & tableRows.group == "all" & tableRows.tolerance == 1.5,:);
    eligible = sortrows(eligible, {'f1','threshold'}, {'descend','descend'});
    selected{end+1} = table2struct(eligible(1,:)); %#ok<AGROW>
end
selected = [selected{:}];
summary = struct2table([rows{:}]);
writetable(summary, fullfile(root,'summary.csv'));
comparison = summary(false(height(summary),1),:);
for k = 1:numel(selected)
    take = summary.model == selected(k).model & ...
        (summary.threshold == selected(k).threshold | summary.is_default == 1);
    comparison = [comparison; summary(take,:)]; %#ok<AGROW>
end
writetable(comparison, fullfile(root,'comparison.csv'));
f = fopen(fullfile(root,'selected.json'),'w');
cleanup = onCleanup(@() fclose(f));
fwrite(f,jsonencode(selected,PrettyPrint=true),'char');
disp(struct2table(selected));
end
