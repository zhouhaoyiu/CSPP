function prepare_revision_figures_matlab()
% Keep all figure statistics traceable to the sealed record-level summaries.
root = 'workspace';
out = fullfile(root,'outputs/cspp_manuscript_revision_20260911/figure_data');
c = readtable(fullfile(root,'outputs/cspp_bu100_threshold_scan_20260911/comparison.csv'),'TextType','string');
publicRoot = fullfile(root,'outputs/cspp_bu100_public_baselines_20260911/valid_segment_origins');
verification = jsondecode(fileread(fullfile(publicRoot,'verification.json')));
assert(strcmp(verification.status,'PASS'));
p = readtable(fullfile(publicRoot,'comparison.csv'),'TextType','string');
selected = jsondecode(fileread(fullfile(publicRoot,'selected.json')));
allRows = {};
for setting = ["best","default"]
    if setting == "best"
        cs = c(c.selection == "best_joint" & c.stratum == "all_records",:);
    else
        cs = c(c.selection == "frozen" & c.stratum == "all_records",:);
    end
    for i = 1:height(cs)
        r = cs(i,:);
        s = struct('method',"CSPP",'setting',setting,'threshold',r.event_threshold, ...
            'candidate_threshold',r.candidate_threshold,'group',r.group,'tolerance',r.tolerance_seconds, ...
            'records',r.records,'tp',r.TP,'fp_proxy',r.FP_proxy,'fn',r.FN, ...
            'precision',r.precision,'recall',r.recall,'f1',r.F1, ...
            'median_absolute_error',r.median_absolute_error_seconds, ...
            'median_availability_latency',r.median_availability_latency_seconds);
        allRows{end+1} = s; %#ok<AGROW>
    end
    for k = 1:numel(selected)
        if setting == "best"
            ps = p(p.model == string(selected(k).model) & p.threshold == selected(k).threshold,:);
        else
            ps = p(p.model == string(selected(k).model) & p.is_default == 1,:);
        end
        ps = ps(ismember(ps.group,["all","strong_motion","intensity"]),:);
        for i = 1:height(ps)
            r = ps(i,:);
            if r.model == "phasenet_instance"
                name = "PhaseNet";
            else
                name = "EQTransformer";
            end
            s = struct('method',name,'setting',setting,'threshold',r.threshold, ...
                'candidate_threshold',NaN,'group',r.group,'tolerance',r.tolerance, ...
                'records',r.records,'tp',r.tp,'fp_proxy',r.fp_proxy,'fn',r.fn, ...
                'precision',r.precision,'recall',r.recall,'f1',r.f1, ...
                'median_absolute_error',r.median_absolute_error, ...
                'median_availability_latency',r.median_availability_latency);
            allRows{end+1} = s; %#ok<AGROW>
        end
    end
end
china = struct2table([allRows{:}]);
assert(height(china) == 54);
assert(all(abs(china.f1 - 2*china.tp./(2*china.tp+china.fp_proxy+china.fn)) < 1e-12));
writetable(china, fullfile(out,'china_comparison.csv'));

source = readtable(fullfile(root,'outputs/cspp_bu100_threshold_scan_20260911/combined_summary.csv'),'TextType','string');
source = source(source.group == "all" & source.stratum == "all_records" & source.tolerance_seconds == 1.5,:);
assert(height(source) == 347);
writetable(source,fullfile(out,'china_cspp_all_thresholds.csv'));
source = readtable(fullfile(publicRoot,'summary.csv'),'TextType','string');
source = source(source.group == "all" & source.tolerance == 1.5,:);
writetable(source,fullfile(out,'china_public_all_thresholds.csv'));

% Matched-record ECDFs retain each method's own matched denominator.
cr = readtable(fullfile(root,'outputs/cspp_bu100_threshold_scan_20260911/refined/selected_record_metrics.csv'),'TextType','string');
cr = cr(cr.selection == "best_joint" & cr.tolerance_seconds == 1.5 & cr.matched == 1,:);
assert(height(cr) == 1931 && all(cr.event_threshold == 0.575));
samples = {abs(cr.error_seconds),cr.latency_seconds};
names = ["CSPP","PhaseNet","EQTransformer"];
keys = ["phasenet_instance","eqtransformer_original"];
expectedMatches = [1979,693];
for k = 1:2
    pr = readtable(fullfile(publicRoot,keys(k),'selected_record_metrics.csv'),'TextType','string');
    pr = pr(pr.tolerance == 1.5 & pr.tp == 1,:);
    assert(height(pr) == expectedMatches(k));
    samples(k+1,:) = {pr.absolute_error,pr.availability_latency};
end
for k = 1:3
    for j = 1:2
        values = sort(samples{k,j});
        assert(all(isfinite(values)));
        xx = repelem(values,2);
        yy = reshape([(0:numel(values)-1);(1:numel(values))],[],1)/numel(values)*100;
        writematrix([xx(:),yy],fullfile(out,sprintf('china_%s_ecdf_%d.txt',names(k),j)),'Delimiter','tab');
    end
end
ab = readtable(fullfile(out,'ablation_all_seeds.csv'),'TextType','string');
arms = unique(ab.arm,'stable');
abRows = cell(numel(arms),1);
for k = 1:numel(arms)
    a = ab(ab.arm == arms(k),:);
    assert(height(a) == 3);
    abRows{k} = struct('arm',arms(k),'precision',median(a.primary_precision), ...
        'recall',median(a.primary_recall),'f1',median(a.primary_f1));
end
writetable(struct2table([abRows{:}]),fullfile(out,'ablation_score_medians.csv'));

example = load(fullfile(root,'paper/tgrs/figures_redrawn_20260904/fig_v27_ncedc_event_examples_source.mat'));
meta = {};
for i = 1:3
    prefix = sprintf('example%d_',i);
    time = example.([prefix,'time']);
    wave = example.([prefix,'wave']);
    writematrix([time(:), wave'],fullfile(out,sprintf('example%d_wave.txt',i)),'Delimiter','tab');
    pt = example.([prefix,'packet_time']);
    keep = pt >= -3 & pt <= 6;
    candidate = example.([prefix,'candidate_probability']);
    event = example.([prefix,'event_probability']);
    writematrix([pt(keep)',candidate(keep)',event(keep)'],fullfile(out,sprintf('example%d_scores.txt',i)),'Delimiter','tab');
    s = struct('index',i,'snr_db',example.([prefix,'snr_db']),'magnitude',example.([prefix,'magnitude']), ...
        'distance_km',example.([prefix,'distance_km']),'pick_error',example.([prefix,'pick_error_seconds']), ...
        'release_delay',example.([prefix,'release_delay_seconds']));
    meta{end+1} = s; %#ok<AGROW>
end
writetable(struct2table([meta{:}]),fullfile(out,'examples_metadata.csv'));
disp(china(china.setting == "best" & china.group == "all" & china.tolerance == 1.5,:));
end
