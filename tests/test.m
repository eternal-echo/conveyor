%% 包络线测试
load('../data/test_data.mat')
processed_signal = abs(signal);
% 计算包络线
[env, ~] = envelope(processed_signal, 50, 'rms');
% 闭运算
se = strel('disk', 120);
env = imclose(env, se);

% 归一化包络线
env = normalizeFeature(env);

% 使用 findpeaks 找到波峰
[pks, locs] = findpeaks(env, 'MinPeakHeight', 0.5);

% 初始化存储波谷的位置
troughs = zeros(length(locs), 2);

% 寻找每个波峰的波谷
search_range = 50;
grad = gradient(env);
for i = 1:length(locs)
    peak_loc = locs(i);
    
    % 寻找左侧波谷
    left_trough = peak_loc;
    while left_trough > search_range && ~all(grad(left_trough-search_range:left_trough) == 0)
        left_trough = left_trough - 1;
    end
    if left_trough == search_range
        left_trough = 1;
    end
    
    % 寻找右侧波谷
    right_trough = peak_loc;
    while right_trough < length(env) - search_range && ~all(grad(right_trough:right_trough+search_range) == 0)
        right_trough = right_trough + 1;
    end
    if right_trough == length(env) - search_range
        right_trough = length(env);
    end
    
    troughs(i, :) = [left_trough, right_trough];
end

% 求包络线的梯度
env_gradient = gradient(env);

% 绘制包络线
figure;
subplot(3, 1, 1);
plot(time, signal);
hold on;
for i = 1:size(troughs, 1)
    fill([time(troughs(i, 1)), time(troughs(i, 2)), time(troughs(i, 2)), time(troughs(i, 1))], [min(signal), min(signal), max(signal), max(signal)], 'r', 'FaceAlpha', 0.3);
end
title('Original Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3, 1, 2);
plot(time, processed_signal);
hold on;
plot(time, env, 'r', 'LineWidth', 2);
hold on;
plot(time(locs), pks, 'go', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
hold on;
for i = 1:size(troughs, 1)
    plot(time(troughs(i, 1)), env(troughs(i, 1)), 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
    hold on;
    plot(time(troughs(i, 2)), env(troughs(i, 2)), 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
    hold on;
    plot([time(troughs(i, 1)), time(troughs(i, 2))], [env(troughs(i, 1)), env(troughs(i, 2))], 'k', 'LineWidth', 2);
end
title('Signal Envelope');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3, 1, 3);
plot(time, env_gradient);
hold on;
for i = 1:size(troughs, 1)
    fill([time(troughs(i, 1)), time(troughs(i, 2)), time(troughs(i, 2)), time(troughs(i, 1))], [min(signal), min(signal), max(signal), max(signal)], 'r', 'FaceAlpha', 0.3);
end
title('Envelope Gradient');
xlabel('Time (s)');
ylabel('Gradient');
grid on;


% Function to normalize a feature
function normalized_feature = normalizeFeature(feature)
    normalized_feature = (feature - min(feature)) / (max(feature) - min(feature));
end



% %% 闭运算 最大最小值滤波测试
% % 生成三角波
% time = linspace(0, 10, 1000); % 时间轴
% combined_feature = sawtooth(2 * pi * 1 * time, 0.5); % 生成三角波
% noise = 0.5 * sawtooth(2 * pi * 2 * time, 0.5);
% % 在noise和combined_feature中取最大值
% combined_feature = max(combined_feature, noise);

% % 绘制三角波
% figure;
% legend('Triangular Wave', 'Noise');
% plot(time, combined_feature);
% % hold on;
% % plot(time, noise);
% title('Triangular Wave');
% xlabel('Time (s)');
% ylabel('Amplitude');
% hold on;

% % 最大值滤波
% window_size = 30
% combined_feature = movmax(combined_feature, window_size);

% % 绘制滤波后的三角波
% figure;
% plot(time, combined_feature);
% title('Triangular Wave After Max Filtering');
% xlabel('Time (s)');
% ylabel('Amplitude');
% grid on;

% % 最小值滤波
% window_size = 30
% combined_feature = movmin(combined_feature, window_size);

% % 绘制滤波后的三角波
% figure;
% plot(time, combined_feature);
% title('Triangular Wave After Min Filtering');
% xlabel('Time (s)');
% ylabel('Amplitude');
% grid on;
% 
% % 闭运算
% se = strel('disk', 30);
% combined_feature = imclose(combined_feature, se);
% 
% % 绘制闭运算后的三角波
% figure;
% plot(time, combined_feature);
% title('Triangular Wave After Closing Operation');
% xlabel('Time (s)');
% ylabel('Amplitude');
% grid on;
% 
% 
% % 寻找波峰
% [peaks, locs, intervals] = detectFeaturesPeaks(combined_feature, time);
% 
% % 绘制波峰
% figure;
% plot(time, combined_feature);
% hold on;
% plot(locs, peaks, 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
% hold on;
% for i = 1:size(intervals, 1)
%     plot([intervals(i, 1), intervals(i, 2)], [0.1, 0.1], 'k', 'LineWidth', 2);
% end
% 
% title('Triangular Wave Peaks and Intervals');
% xlabel('Time (s)');
% ylabel('Amplitude');
% grid on;
% 
% 
% % Function to detect peaks and intervals
% function [peaks, locs, merged_intervals] = detectFeaturesPeaks(combined_feature, time)
%     [peaks, locs, ~, ~] = findpeaks(combined_feature, time, 'MinPeakHeight', 0.05, 'MinPeakDistance', 0.005, 'MinPeakProminence', 0.05);
%     [~, locs_idx] = ismember(locs, time);
%     height_threshold = 0.3;
%     peak_intervals = getPeakIntervals(locs_idx, peaks, combined_feature, time, height_threshold);
%     merged_intervals = mergeIntervals(peak_intervals);
% end
% 
% % Function to get peak intervals
% function peak_intervals = getPeakIntervals(locs_idx, peaks, combined_feature, time, height_threshold)
%     left_ips = zeros(1, length(locs_idx));
%     right_ips = zeros(1, length(locs_idx));
%     for i = 1:length(locs_idx)
%         left_base = locs_idx(i);
%         while left_base > 1 && combined_feature(left_base) > height_threshold * peaks(i)
%             left_base = left_base - 1;
%         end
%         left_ips(i) = time(left_base);
% 
%         right_base = locs_idx(i);
%         while right_base < length(time) && combined_feature(right_base) > height_threshold * peaks(i)
%             right_base = right_base + 1;
%         end
%         right_ips(i) = time(right_base);
%     end
%     peak_intervals = [left_ips; right_ips]';
% end
% 
% % Function to merge intervals
% function merged_intervals = mergeIntervals(peak_intervals)
%     merged_intervals = [];
%     if ~isempty(peak_intervals)
%         peak_intervals = sortrows(peak_intervals);
%         current_interval = peak_intervals(1, :);
%         for i = 2:size(peak_intervals, 1)
%             if peak_intervals(i, 1) <= current_interval(2)
%                 current_interval(2) = max(current_interval(2), peak_intervals(i, 2));
%             else
%                 merged_intervals = [merged_intervals; current_interval];
%                 current_interval = peak_intervals(i, :);
%             end
%         end
%         merged_intervals = [merged_intervals; current_interval];
%     end
% end
