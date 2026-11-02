function aggregate_component_gate_ablation(output)
% Pair gate policies on identical records; keep boundary proximity separate.
H = readtable(fullfile(output,'record_health.csv'),TextType='string',VariableNamingRule='preserve');
C = readtable(fullfile(output,'candidates.csv'),TextType='string',VariableNamingRule='preserve');
B = readtable(fullfile(output,'boundaries.csv'),TextType='string',VariableNamingRule='preserve');
[group,names] = findgroups(C.id + "|" + C.variant + "|" + C.case);
indices = splitapply(@(x){x},(1:height(C))',group);
lookup = containers.Map(cellstr(names),indices);
assert(height(H)==645*2*4);
H.tp = zeros(height(H),1);
H.fp_reference = nan(height(H),1);
H.fn = nan(height(H),1);
H.pick_error_seconds = nan(height(H),1);
H.first_delay_seconds = nan(height(H),1);
H.start_edge_candidates = zeros(height(H),1);
H.end_edge_candidates = zeros(height(H),1);
H.estimated_start_edge_candidates = zeros(height(H),1);
H.estimated_end_edge_candidates = zeros(height(H),1);
H.matched_near_start_edge = zeros(height(H),1);
for k=1:height(H)
    if H.kind(k)=="event"
        H.fp_reference(k) = H.candidate_count(k);
        H.fn(k) = 1;
    end
    key = char(H.id(k)+"|"+H.variant(k)+"|"+H.case(k));
    if ~isKey(lookup,key), continue; end
    ix = lookup(key);
    bounds = B(B.id==H.id(k),:);
    available = C.available_epoch(ix);
    estimated = C.estimated_p_epoch(ix);
    start_available = any(available>=bounds.start_epoch' & available<=bounds.start_epoch'+2,2);
    end_available = any(available>=bounds.end_epoch' & available<=bounds.end_epoch'+2,2);
    start_estimated = any(abs(estimated-bounds.start_epoch')<=0.5,2);
    end_estimated = any(abs(estimated-bounds.end_epoch')<=0.5,2);
    H.start_edge_candidates(k) = sum(start_available);
    H.end_edge_candidates(k) = sum(end_available);
    H.estimated_start_edge_candidates(k) = sum(start_estimated);
    H.estimated_end_edge_candidates(k) = sum(end_estimated);
    if H.kind(k)~="event", continue; end
    errors = estimated-H.reference_p_epoch(k);
    good = find(abs(errors)<=1.5);
    if isempty(good), continue; end
    [~,order] = sortrows([abs(errors(good)),available(good)],[1,2]);
    H.tp(k) = 1;
    H.fn(k) = 0;
    H.fp_reference(k) = H.candidate_count(k)-1;
    H.pick_error_seconds(k) = errors(good(order(1)));
    H.first_delay_seconds(k) = min(available(good))-H.reference_p_epoch(k);
    H.matched_near_start_edge(k) = any(start_estimated(good));
end
writetable(H,fullfile(output,'record_metrics.csv'));
variants = ["original_gate","two_component_gate"];
cases = ["clean","zero_Z","zero_N","zero_E"];
summaries = {}; backgrounds = {};
original_clean = sortrows(H(H.kind=="event" & H.variant==variants(1) & H.case=="clean",:),'id');
for v=1:2
    clean = sortrows(H(H.kind=="event" & H.variant==variants(v) & H.case=="clean",:),'id');
    clean_bg = sortrows(H(H.kind=="background" & H.variant==variants(v) & H.case=="clean",:),'id');
    assert(height(clean)==633 && height(clean_bg)==12);
    [event_group,events] = findgroups(clean.event_id);
    assert(numel(events)==65);
    event_n = accumarray(event_group,ones(633,1));
    rng(20260908,'twister'); draws = randi(65,65,2000);
    for j=1:4
        r = sortrows(H(H.kind=="event" & H.variant==variants(v) & H.case==cases(j),:),'id');
        old = sortrows(H(H.kind=="event" & H.variant==variants(1) & H.case==cases(j),:),'id');
        assert(height(r)==633 && isequal(r.id,clean.id) && isequal(r.id,old.id));
        tp=sum(r.tp); fp=sum(r.fp_reference); fn=sum(r.fn);
        event_delta=accumarray(event_group,r.tp-old.tp);
        boot=100*sum(event_delta(draws),1)./sum(event_n(draws),1);
        ci=prctile(boot,[2.5,97.5]);
        summaries{end+1}=struct('variant',variants(v),'case',cases(j),'records',633,'events',65, ...
            'tp',tp,'fp_reference',fp,'fn',fn,'precision',divide(tp,tp+fp),'recall',tp/633, ...
            'f1',divide(2*tp,2*tp+fp+fn), ...
            'tp_change_from_original_gate_same_case',tp-sum(old.tp), ...
            'fp_change_from_original_gate_same_case',fp-sum(old.fp_reference), ...
            'recall_change_from_original_gate_same_case_pp',100*(tp-sum(old.tp))/633, ...
            'recall_change_ci_low_pp',ci(1),'recall_change_ci_high_pp',ci(2), ...
            'lost_from_original_clean',sum(original_clean.tp==1 & r.tp==0), ...
            'gained_from_original_clean',sum(original_clean.tp==0 & r.tp==1), ...
            'absolute_error_median_s',q(abs(r.pick_error_seconds),50), ...
            'absolute_error_p95_s',q(abs(r.pick_error_seconds),95), ...
            'first_delay_median_s',q(r.first_delay_seconds,50), ...
            'first_delay_p95_s',q(r.first_delay_seconds,95), ...
            'recall_within_2s',sum(r.tp==1 & r.first_delay_seconds<=2)/633, ...
            'mean_allowed_seconds',mean(r.allowed_seconds), ...
            'mean_disallowed_during_fault_seconds',mean(r.disallowed_during_fault_seconds), ...
            'recovery_median_s',q(r.recovery_mean_seconds,50),'recovery_p95_s',q(r.recovery_mean_seconds,95), ...
            'accepted_zero_packets',sum(r.accepted_zero_packets), ...
            'start_edge_candidates',sum(r.start_edge_candidates),'end_edge_candidates',sum(r.end_edge_candidates), ...
            'matched_near_start_edge_records',sum(r.matched_near_start_edge)); %#ok<AGROW>
        b=sortrows(H(H.kind=="background" & H.variant==variants(v) & H.case==cases(j),:),'id');
        old_b=sortrows(H(H.kind=="background" & H.variant==variants(1) & H.case==cases(j),:),'id');
        assert(height(b)==12 && isequal(b.id,clean_bg.id));
        hours=sum(b.duration_seconds)/3600; assert(hours==12);
        backgrounds{end+1}=struct('variant',variants(v),'case',cases(j),'station_hours',hours, ...
            'candidate_count',sum(b.candidate_count),'candidates_per_station_hour',sum(b.candidate_count)/hours, ...
            'candidate_delta_from_original_gate',sum(b.candidate_count-old_b.candidate_count), ...
            'candidate_delta_from_same_variant_clean',sum(b.candidate_count-clean_bg.candidate_count), ...
            'mean_allowed_seconds',mean(b.allowed_seconds), ...
            'mean_disallowed_during_fault_seconds',mean(b.disallowed_during_fault_seconds), ...
            'recovery_median_s',q(b.recovery_mean_seconds,50),'accepted_zero_packets',sum(b.accepted_zero_packets), ...
            'start_edge_candidates',sum(b.start_edge_candidates),'end_edge_candidates',sum(b.end_edge_candidates), ...
            'estimated_start_edge_candidates',sum(b.estimated_start_edge_candidates), ...
            'estimated_end_edge_candidates',sum(b.estimated_end_edge_candidates)); %#ok<AGROW>
    end
end
S=struct2table([summaries{:}]); G=struct2table([backgrounds{:}]);
writetable(S,fullfile(output,'event_summary.csv'));
writetable(G,fullfile(output,'background_summary.csv'));
result=struct('event',[summaries{:}],'background',[backgrounds{:}], ...
    'matlab_version',version,'scope','Frozen-weight gate-only diagnostic; not a validated production fix', ...
    'edge_definition','Available time in [boundary,boundary+2s]; estimated time within +/-0.5s; clean rows use paired reference positions', ...
    'caution','Background may contain real arrivals. Event fault onset is tied to reference P, so boundary-coincident matches need caution.');
fid=fopen(fullfile(output,'results.json'),'w');
fprintf(fid,'%s\n',jsonencode(result,PrettyPrint=true)); fclose(fid);
disp(S(:,{'variant','case','tp','fp_reference','recall','f1'}));
disp(G(:,{'variant','case','candidate_count','start_edge_candidates','end_edge_candidates'}));
end

function value=divide(a,b)
if b==0, value=NaN; else, value=a/b; end
end

function value=q(x,p)
x=x(isfinite(x));
if isempty(x), value=NaN; else, value=prctile(x,p); end
end
