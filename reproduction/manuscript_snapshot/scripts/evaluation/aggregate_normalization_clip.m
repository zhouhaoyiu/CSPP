function aggregate_normalization_clip(output)
% Paired event-cluster inference; all records and conditions remain included.
P=jsondecode(fileread(fullfile(output,'protocol.json')));
H=readtable(fullfile(output,'record_health.csv'),TextType='string',VariableNamingRule='preserve');
C=readtable(fullfile(output,'candidates.csv'),TextType='string',VariableNamingRule='preserve');
L=readtable(fullfile(output,'clipping_counts.csv'),TextType='string',VariableNamingRule='preserve');
assert(height(H)==645*4);
[group,names]=findgroups(C.id+"|"+C.case);
indices=splitapply(@(x){x},(1:height(C))',group);
lookup=containers.Map(cellstr(names),indices);
H.tp=zeros(height(H),1); H.tp_05=H.tp; H.tp_025=H.tp;
H.fp_reference=nan(height(H),1); H.fn=H.fp_reference;
H.pick_error_seconds=H.fp_reference; H.first_delay_seconds=H.fp_reference;
for k=1:height(H)
    if H.kind(k)~="event", continue; end
    H.fp_reference(k)=H.candidate_count(k); H.fn(k)=1;
    key=char(H.id(k)+"|"+H.case(k));
    if ~isKey(lookup,key), continue; end
    ix=lookup(key); errors=C.estimated_p_epoch(ix)-H.reference_p_epoch(k);
    available=C.available_epoch(ix);
    H.tp_05(k)=any(abs(errors)<=0.5); H.tp_025(k)=any(abs(errors)<=0.25);
    good=find(abs(errors)<=1.5);
    if isempty(good), continue; end
    [~,order]=sortrows([abs(errors(good)),available(good)],[1,2]);
    H.tp(k)=1; H.fn(k)=0; H.fp_reference(k)=H.candidate_count(k)-1;
    H.pick_error_seconds(k)=errors(good(order(1)));
    H.first_delay_seconds(k)=min(available(good))-H.reference_p_epoch(k);
end
writetable(H,fullfile(output,'record_metrics.csv'));
cases=string(P.cases)'; fields=["tp","tp_05","tp_025"]; tolerances=[1.5,0.5,0.25];
clean=sortrows(H(H.kind=="event" & H.case=="clip12",:),'id');
assert(height(clean)==633);
[event_group,events]=findgroups(clean.event_id); assert(numel(events)==65);
event_n=accumarray(event_group,ones(height(clean),1));
rng(P.evaluation.bootstrap.seed,'twister'); draws=randi(65,65,P.evaluation.bootstrap.draws);
save(fullfile(output,'bootstrap_draws.mat'),'draws','events','event_n');
summaries={}; backgrounds={}; sensitivities={}; clippings={};
for j=1:4
    r=sortrows(H(H.kind=="event" & H.case==cases(j),:),'id');
    assert(isequal(r.id,clean.id));
    assert(isequal(r.allowed_seconds,clean.allowed_seconds) && isequal(r.invalid_seconds,clean.invalid_seconds));
    for t=1:3
        delta=r.(fields(t))-clean.(fields(t)); event_delta=accumarray(event_group,delta);
        boot=100*sum(event_delta(draws),1)./sum(event_n(draws),1); ci=prctile(boot,[2.5,97.5]);
        sensitivity=struct('case',cases(j),'tolerance_s',tolerances(t),'records',633,'events',65, ...
            'tp',sum(r.(fields(t))),'recall',mean(r.(fields(t))), ...
            'gain',sum(clean.(fields(t))==0 & r.(fields(t))==1),'loss',sum(clean.(fields(t))==1 & r.(fields(t))==0), ...
            'recall_delta_pp',100*mean(delta),'recall_delta_ci_low_pp',ci(1),'recall_delta_ci_high_pp',ci(2));
        sensitivities{end+1}=sensitivity; %#ok<AGROW>
        if t==1, primary=sensitivity; end
    end
    tp=sum(r.tp); fp=sum(r.fp_reference); fn=sum(r.fn); common=r.tp==1 & clean.tp==1;
    summaries{end+1}=struct('case',cases(j),'records',633,'events',65,'tp',tp,'fp_reference',fp,'fn',fn, ...
        'precision',tp/(tp+fp),'recall',tp/633,'f1',2*tp/(2*tp+fp+fn), ...
        'gain',primary.gain,'loss',primary.loss,'recall_delta_pp',primary.recall_delta_pp, ...
        'recall_delta_ci_low_pp',primary.recall_delta_ci_low_pp,'recall_delta_ci_high_pp',primary.recall_delta_ci_high_pp, ...
        'absolute_error_median_s',q(abs(r.pick_error_seconds),50),'absolute_error_p95_s',q(abs(r.pick_error_seconds),95), ...
        'first_delay_median_s',q(r.first_delay_seconds,50),'first_delay_p95_s',q(r.first_delay_seconds,95), ...
        'common_hit_records',sum(common),'paired_delay_delta_median_s',q(r.first_delay_seconds(common)-clean.first_delay_seconds(common),50), ...
        'paired_absolute_error_delta_median_s',q(abs(r.pick_error_seconds(common))-abs(clean.pick_error_seconds(common)),50)); %#ok<AGROW>
    b=H(H.kind=="background" & H.case==cases(j),:); assert(height(b)==12 && sum(b.duration_seconds)==43200);
    backgrounds{end+1}=struct('case',cases(j),'station_hours',12,'candidate_count',sum(b.candidate_count), ...
        'candidates_per_station_hour',sum(b.candidate_count)/12,'allowed_station_hours',sum(b.allowed_seconds)/3600); %#ok<AGROW>
    for kind=["event","background"]
        subset=L(L.case==cases(j) & L.kind==kind,:);
        for scope=unique(subset.scope)'
            a=subset(subset.scope==scope,:);
            clippings{end+1}=struct('case',cases(j),'kind',kind,'scope',scope,'records',height(a), ...
                'sample_total',sum(a.sample_total),'clipped_total',sum(a.clipped_total), ...
                'clipped_fraction',sum(a.clipped_total)/sum(a.sample_total), ...
                'records_with_clipping',sum(a.clipped_total>0), ...
                'clipped_Z',sum(a.clipped_Z),'clipped_N',sum(a.clipped_N),'clipped_E',sum(a.clipped_E), ...
                'maximum_absolute_preclip',max(a.max_absolute_preclip)); %#ok<AGROW>
        end
    end
end
S=struct2table([summaries{:}]); B=struct2table([backgrounds{:}]); T=struct2table([sensitivities{:}]); L=struct2table([clippings{:}]);
writetable(S,fullfile(output,'event_summary.csv')); writetable(B,fullfile(output,'background_summary.csv'));
writetable(T,fullfile(output,'tolerance_summary.csv')); writetable(L,fullfile(output,'clipping_summary.csv'));
result=struct('event',[summaries{:}],'background',[backgrounds{:}],'tolerances',[sensitivities{:}], ...
    'clipping',[clippings{:}],'matlab_version',version,'primary_tolerance_s',1.5, ...
    'bootstrap','10000 paired event-cluster draws against clip12; percentile 95% intervals; exploratory, no multiplicity adjustment');
fid=fopen(fullfile(output,'results.json'),'w'); fprintf(fid,'%s\n',jsonencode(result,PrettyPrint=true)); fclose(fid);
disp(S(:,{'case','tp','fp_reference','recall','f1','gain','loss','recall_delta_ci_low_pp','recall_delta_ci_high_pp'}));
disp(B); disp(T(:,{'case','tolerance_s','tp'}));
end

function value=q(x,p)
x=x(isfinite(x));
if isempty(x), value=NaN; else, value=prctile(x,p); end
end
