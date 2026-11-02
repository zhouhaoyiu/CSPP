function compare_bu100_rebased_matlab()
% Compare identical populations and frozen threshold grids before/after rebase.
root = 'workspace/outputs/cspp_bu100_public_baselines_20260911';
out = fullfile(root, 'valid_segment_origins');
before = readtable(fullfile(root,'refined','summary.csv'),'TextType','string');
after = readtable(fullfile(out,'summary.csv'),'TextType','string');
oldCoverage = readtable(fullfile(root,'reference_coverage.csv'),'TextType','string');
newCoverage = readtable(fullfile(out,'reference_coverage.csv'),'TextType','string');
results = {};
for model = ["phasenet_instance", "eqtransformer_original"]
    previous = before(before.model == model & before.group == "all" & before.tolerance == 1.5,:);
    previous = sortrows(previous,{'f1','threshold'},{'descend','descend'});
    current = after(after.model == model & after.group == "all" & after.tolerance == 1.5,:);
    current = sortrows(current,{'f1','threshold'},{'descend','descend'});
    for group = ["all", "strong_motion", "intensity"]
        a = before(before.model == model & before.group == group & before.tolerance == 1.5 & before.threshold == previous.threshold(1),:);
        b = after(after.model == model & after.group == group & after.tolerance == 1.5 & after.threshold == current.threshold(1),:);
        fixed = after(after.model == model & after.group == group & after.tolerance == 1.5 & after.threshold == previous.threshold(1),:);
        x = oldCoverage(oldCoverage.model == model & (group == "all" | oldCoverage.instrument == group),:);
        y = newCoverage(newCoverage.model == model & (group == "all" | newCoverage.instrument == group),:);
        assert(isequal(x.id,y.id));
        s = struct('model',model,'group',group,'records',height(x), ...
            'old_threshold',a.threshold,'new_threshold',b.threshold, ...
            'old_f1',a.f1,'new_f1',b.f1,'new_f1_at_old_threshold',fixed.f1, ...
            'old_tp',a.tp,'new_tp',b.tp,'old_fp_proxy',a.fp_proxy,'new_fp_proxy',b.fp_proxy, ...
            'old_reference_covered',sum(x.reference_sample_published), ...
            'new_reference_covered',sum(y.reference_sample_published), ...
            'old_any_tolerance_covered',sum(x.any_sample_within_tolerance_published), ...
            'new_any_tolerance_covered',sum(y.any_sample_within_tolerance_published), ...
            'old_no_windows',nnz(x.native_windows == 0),'new_no_windows',nnz(y.native_windows == 0));
        results{end+1} = s; %#ok<AGROW>
    end
end
comparison = struct2table([results{:}]);
writetable(comparison,fullfile(out,'before_after.csv'));
disp(comparison);
end
