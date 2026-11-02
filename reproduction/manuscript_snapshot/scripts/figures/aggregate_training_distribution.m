function aggregate_training_distribution(output)
% Join frozen record indices to provider metadata and create GMT-ready data.
P=jsondecode(fileread(fullfile(output,'protocol.json')));
I=load(fullfile(output,'selected_indices.mat'));
base_fields=["source_magnitude","source_depth_km","source_latitude_deg","source_longitude_deg"];
ig_fields=[base_fields,"source_id","trace_name","split","source_magnitude_type","path_ep_distance_km", ...
    "station_code","station_latitude_deg","station_longitude_deg","source_origin_time"];
ig=load_fields(P.sources.instancegm_metadata.path,ig_fields);
ig=ig(double(I.ig_train(:))+1,:);
assert(all(ig.split=="train"));
A=canonical(ig,"InstanceGM",double(I.ig_train(:)),ig.source_id,ig.source_magnitude_type,ig.path_ep_distance_km);
clear ig;
kn_fields=[base_fields,"event_name","trace_name","split","source_distance_km","station_code", ...
    "station_latitude_deg","station_longitude_deg"];
kn=load_fields(P.sources.knet_metadata.path,kn_fields);
kn=kn(double(I.knet_train(:))+1,:); assert(all(kn.split=="train"));
B=canonical(kn,"K-NET",double(I.knet_train(:)),kn.event_name,repmat("unspecified",height(kn),1),kn.source_distance_km);
clear kn;
store_fields=["store_index","dataset_index","trace_name","event_id","station_id","split","source_magnitude","source_distance_km"];
store=load_fields(P.sources.cwa_store_metadata.path,store_fields);
store=store(double(I.cwa2020_opened_train(:))+1,:);
assert(all(store.split=="train") && isequal(store.store_index,double(I.cwa2020_opened_train(:))));
cwa_fields=[base_fields,"source_event_id","trace_name","source_magnitude_type","path_ep_distance_km", ...
    "station_code","station_latitude_deg","station_longitude_deg","source_origin_time"];
cwa=load_fields(P.sources.cwa_source_metadata.path,cwa_fields);
cwa=cwa(store.dataset_index+1,:);
assert(isequal(cwa.trace_name,store.trace_name));
assert(isequal(cwa.source_event_id,store.event_id));
assert(max(abs(cwa.source_magnitude-store.source_magnitude))<1e-9);
assert(max(abs(cwa.path_ep_distance_km-store.source_distance_km))<1e-6);
C=canonical(cwa,"CWA",double(I.cwa2020_opened_train(:)),store.event_id,cwa.source_magnitude_type,store.source_distance_km);
C.original_dataset_index=store.dataset_index;
A.original_dataset_index=A.source_index; B.original_dataset_index=B.source_index;
R=[A;B;C]; assert(height(R)==2500 && all(R.magnitude>=3) && all(R.epicentral_distance_km<=200));
writetable(R,fullfile(output,'selected_training_records.csv'));
labels=["InstanceGM","K-NET","CWA"]; slugs=["instancegm","knet","cwa"];
expected_counts=[1000,1000,500]; magnitude_upper=max(6.6,ceil(max(R.magnitude)/0.2)*0.2);
summary={}; events_all=table(); conflicts_all=table();
for k=1:3
    r=R(R.dataset==labels(k),:);
    assert(height(r)==expected_counts(k));
    [g,ids]=findgroups(r.event_id); n=numel(ids);
    event=table(repmat(labels(k),n,1),ids,zeros(n,1),nan(n,1),nan(n,1),nan(n,1),nan(n,1), ...
        'VariableNames',{'dataset','event_id','record_count','longitude','latitude','magnitude','depth_km'});
    conflict=table();
    for j=1:n
        part=r(g==j,:); event.record_count(j)=height(part);
        for field=["longitude","latitude","magnitude","depth_km"]
            values=part.(field); values=values(isfinite(values));
            if ~isempty(values), event.(field)(j)=values(1); end
            if ~isempty(values) && max(values)-min(values)>1e-6
                row=table(labels(k),ids(j),field,min(values),max(values), ...
                    'VariableNames',{'dataset','event_id','field','minimum','maximum'});
                conflict=[conflict;row]; %#ok<AGROW>
            end
        end
    end
    conflicts_all=[conflicts_all;conflict]; %#ok<AGROW>
    events_all=[events_all;event]; %#ok<AGROW>
    valid_map=isfinite(event.longitude) & isfinite(event.latitude);
    assert(all(abs(event.longitude(valid_map))<=180) && all(abs(event.latitude(valid_map))<=90));
    writematrix([event.longitude(valid_map),event.latitude(valid_map)],fullfile(output,slugs(k)+"_epicenters.txt"),'Delimiter','tab');
    stats=struct('dataset',labels(k),'slug',slugs(k),'record_count',height(r),'event_count',n, ...
        'mapped_events',sum(valid_map),'events_missing_coordinates',sum(~valid_map), ...
        'station_count',numel(unique(r.station_id)),'metadata_conflicts',height(conflict), ...
        'magnitude_types',{cellstr(unique(r.magnitude_type))}, ...
        'longitude_range',[min(event.longitude(valid_map)),max(event.longitude(valid_map))], ...
        'latitude_range',[min(event.latitude(valid_map)),max(event.latitude(valid_map))]);
    for field=["magnitude","depth_km"]
        values=r.(field); valid=values(isfinite(values));
        assert(~isempty(valid));
        if field=="magnitude"
            step=0.2; lower=3; upper=magnitude_upper;
        else
            maximum=max(valid);
            if maximum<=60, step=2; elseif maximum<=200, step=10; else, step=20; end
            lower=min(0,floor(min(valid)/step)*step); upper=ceil(maximum/step)*step;
        end
        edges=lower:step:upper;
        if edges(end)<max(valid), edges(end+1)=edges(end)+step; end
        bins=histcounts(valid,edges); assert(sum(bins)==numel(valid));
        centers=(edges(1:end-1)+edges(2:end))/2;
        writematrix([centers(:),bins(:)],fullfile(output,slugs(k)+"_"+field+"_histogram.txt"),'Delimiter','tab');
        detail=struct('valid_records',numel(valid),'missing_records',sum(~isfinite(values)), ...
            'minimum',min(valid),'maximum',max(valid),'mean',mean(valid),'median',median(valid), ...
            'bin_width',step,'edges',edges,'counts',bins,'x_limits',[edges(1),edges(end)]);
        stats.(field)=detail;
    end
    summary{end+1}=stats; %#ok<AGROW>
end
writetable(events_all,fullfile(output,'unique_training_events.csv'));
if ~isempty(conflicts_all), writetable(conflicts_all,fullfile(output,'event_metadata_conflicts.csv')); end
result=struct('datasets',[summary{:}],'matlab_version',version,'total_records',height(R), ...
    'histogram_grain','selected waveform records','map_grain','unique events within each source', ...
    'magnitude_scale_note','Provider magnitude values retained; types are recorded and no conversion to Mw is applied', ...
    'metadata_conflicts',height(conflicts_all));
fid=fopen(fullfile(output,'distribution_summary.json'),'w'); fprintf(fid,'%s\n',jsonencode(result,PrettyPrint=true)); fclose(fid);
disp(jsonencode(result,PrettyPrint=true));
assert(isempty(conflicts_all),'Source event metadata conflicts require review before plotting.');
end

function T=load_fields(path,fields)
opts=detectImportOptions(path,'VariableNamingRule','preserve');
assert(all(ismember(fields,string(opts.VariableNames))));
opts.SelectedVariableNames=cellstr(fields);
numeric=["source_magnitude","source_depth_km","source_latitude_deg","source_longitude_deg", ...
    "path_ep_distance_km","source_distance_km","station_latitude_deg","station_longitude_deg","store_index","dataset_index"];
numeric=intersect(fields,numeric,'stable');
text=setdiff(fields,numeric,'stable');
opts=setvartype(opts,cellstr(numeric),'double'); opts=setvartype(opts,cellstr(text),'string');
T=readtable(path,opts);
end

function T=canonical(S,name,source_index,event_id,magnitude_type,distance)
T=table(repmat(name,height(S),1),source_index,string(S.trace_name),string(event_id), ...
    S.source_longitude_deg,S.source_latitude_deg,S.source_magnitude,S.source_depth_km, ...
    string(magnitude_type),distance,string(S.station_code),S.station_longitude_deg,S.station_latitude_deg, ...
    'VariableNames',{'dataset','source_index','trace_name','event_id','longitude','latitude','magnitude','depth_km', ...
    'magnitude_type','epicentral_distance_km','station_id','station_longitude','station_latitude'});
assert(~any(ismissing(T.event_id)) && numel(unique(T.source_index))==height(T));
end
