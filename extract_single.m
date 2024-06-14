addpath('src');
addpath('data');
data_path = 'data';
result_path = 'results';

% Load the data
load('test_data.mat');

%% 1. 去噪和基线漂移校正
denoised_signal = SignalExtract.denoiseSignal(signal, time);

% Plot denoised signal
plot(time, denoised_signal);
title('去噪信号');
xlabel('时间 (秒)');
ylabel('幅度');

%% 2. 提取特征
windowSize = 40;
hopSize = 1;
[energy, autocorr, autocorr1D, combined_feature] = SignalExtract.extractFeatures(denoised_signal, time, 'windowSize', windowSize, 'hopSize', hopSize);

% Plot short-time energy and short-time autocorrelation
figure;
subplot(2, 1, 1);
plot(time, energy);
title('短时能量');
xlabel('时间 (秒)');
ylabel('能量');

subplot(2, 1, 2);
imagesc(time, 1:windowSize, autocorr);
title('短时自相关');
xlabel('时间 (秒)');
ylabel('滞后');
colorbar;

%% 3. 对特征进行峰值检测和间隔合并
peak_boundaries_ratio = 0.3;
[peaks, locs, locs_idx, merged_intervals] = SignalExtract.detectFeaturesPeaks(combined_feature, time, 'peakBoundariesRatio', peak_boundaries_ratio);

% 绘制特征和波峰区间
SignalExtract.plotSignalWithIntervals(time, combined_feature, merged_intervals, 'title', '加权求和特征及其波峰区间', 'markers', [locs; peaks]);

% Plot denoised signal with peak intervals
SignalExtract.plotSignalWithIntervals(time, denoised_signal, merged_intervals, 'title', '去噪信号及其波峰区间');

%% 4. 有效区间检测
envelope_signal = SignalExtract.extractEnvelope(signal);
[valid_peaks_idx, valid_intervals] = SignalExtract.detectValidIntervals(envelope_signal, time, locs_idx, merged_intervals);


valid_count = 0;
data_loader.dataset_df.ValidCount = -1 * ones(height(data_loader.dataset_df), 1);

% 保存结果
for k = 1:size(valid_intervals, 2)
    valid_count = valid_count + 1;
    start_time = valid_intervals(1, k);
    end_time = valid_intervals(2, k);
    mask = (data_loader.dataset_df.RelativeTime >= start_time) & (data_loader.dataset_df.RelativeTime <= end_time) & (data_loader.dataset_df.WorkpieceID == workpiece_id) & (data_loader.dataset_df.MeasurementID == measurement_id);
    data_loader.dataset_df.ValidCount(mask) = valid_count;
end

% 保存数据集
writetable(data_loader.dataset_df, fullfile(result_path, 'dataset_valid.csv'));

% 获取有效区间
valid_intervals = data_loader.getValidIntervals(workpiece_id, measurement_id);

SignalExtract.plotSignalWithIntervals(time, envelope_signal, valid_intervals, 'title', '包络线及其有效区间', 'markers', [time(valid_peaks_idx); envelope_signal(valid_peaks_idx)]);

SignalExtract.plotSignalWithIntervals(time, signal, valid_intervals, 'title', '原始信号及其有效区间', 'markers', [time(valid_peaks_idx); signal(valid_peaks_idx)]);
