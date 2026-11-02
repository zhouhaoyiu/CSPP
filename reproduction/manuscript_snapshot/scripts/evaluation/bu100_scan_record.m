function [count, matched, errors, latencies] = bu100_scan_record(scores, reference, qs, es, tolerances)
% Vectorized four-packet support, clearing all history at invalid packets.
scores = double(scores);
valid = scores(:, 4) ~= 0;
eligible = scores(:, 1);
eligible(~valid | scores(:, 5) == 0) = -Inf;
n = size(scores, 1);
recent = eligible;
bridge = valid;
for lag = 1:min(3, n - 1)
    bridge = bridge & [false(lag, 1); valid(1:end-lag)];
    shifted = [-Inf(lag, 1); eligible(1:end-lag)];
    shifted(~bridge) = -Inf;
    recent = max(recent, shifted);
end
active = valid & (recent >= qs) & (scores(:, 2) >= es);
emitted = active & ~[false(1, numel(qs)); active(1:end-1, :)];
count = sum(emitted, 1);
available = (1:n)' * 0.5;
estimated = available - scores(:, 3) * 2;
matched = false(numel(qs), numel(tolerances));
errors = NaN(numel(qs), numel(tolerances));
latencies = errors;
for k = 1:numel(tolerances)
    mask = emitted & abs(estimated - reference) <= tolerances(k);
    [hit, first] = max(mask, [], 1);
    matched(:, k) = hit';
    e = estimated(first) - reference; d = available(first) - reference;
    e(~hit) = NaN; d(~hit) = NaN;
    errors(:, k) = e(:); latencies(:, k) = d(:);
end
end
