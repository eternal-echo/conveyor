addpath('src');
addpath('data');
data_path = 'data';
result_path = 'results';
extract_path = fullfile(result_path, 'extract');
% 创建保存结果的文件夹
if ~exist(extract_path, 'dir')
    mkdir(extract_path);
end

% 加载数据集
data_loader = DataLoader(fullfile(data_path, 'dataset.csv'));
data_loader.dataset_df.ValidCount = -1 * ones(height(data_loader.dataset_df), 1);

% 遍历每个工件和测量
workpiece_ids = data_loader.getWorkpieceIDs();
for i = 1:length(workpiece_ids)
    workpiece_id = workpiece_ids(i);
    measurement_ids = data_loader.getMeasurementIDs(workpiece_id);

    valid_count = 0;
    for j = 1:length(measurement_ids)
        measurement_id = measurement_ids(j);
        data = data_loader.getMeasurementData(workpiece_id, measurement_id);
        time = data.RelativeTime;
        signal = data.CH1;

        %% 1. 去噪和基线漂移校正
        denoised_signal = SignalExtract.denoiseSignal(signal, time);
        
        %% 2. 提取特征
        windowSize = 40;
        hopSize = 1;
        [energy, autocorr, autocorr1D, combined_feature] = SignalExtract.extractFeatures(denoised_signal, time, 'windowSize', windowSize, 'hopSize', hopSize);
        
        %% 3. 对特征进行峰值检测和间隔合并
        peak_boundaries_ratio = 0.3;
        [peaks, locs, locs_idx, intervals] = SignalExtract.detectFeaturesPeaks(combined_feature, time, 'peakBoundariesRatio', peak_boundaries_ratio);
        
        %% 4. 有效区间检测
        envelope_signal = SignalExtract.extractEnvelope(signal);
        [valid_peaks_idx, valid_intervals] = SignalExtract.detectValidIntervals(envelope_signal, time, locs_idx, intervals);

        %% 保存结果
        for k = 1:size(valid_intervals, 2)
            valid_count = valid_count + 1;
            start_time = valid_intervals(1, k);
            end_time = valid_intervals(2, k);
            mask = (data_loader.dataset_df.RelativeTime >= start_time) & (data_loader.dataset_df.RelativeTime <= end_time) & (data_loader.dataset_df.WorkpieceID == workpiece_id) & (data_loader.dataset_df.MeasurementID == measurement_id);
            data_loader.dataset_df.ValidCount(mask) = valid_count;
        end

        %% 可视化信号和有效区间
        figure('Visible', 'off'); % 不显示图像
        plot(time, signal);
        title(['Workpiece ', num2str(workpiece_id), ', Measurement ', num2str(measurement_id)]);
        xlabel('Time (s)');
        ylabel('Amplitude');
        grid on;
        hold on;
        % 绘制有效区间
        for k = 1:size(valid_intervals, 2)
            start_time = valid_intervals(1, k);
            end_time = valid_intervals(2, k);
            fill([start_time, end_time, end_time, start_time], [min(signal), min(signal), max(signal), max(signal)], 'r', 'FaceAlpha', 0.3);
        end
        % 保存图像
        saveas(gcf, fullfile(extract_path, ['workpiece_', num2str(workpiece_id), '_measurement_', num2str(measurement_id), '.png']));
        close(gcf);
    end
end

% 保存数据集
writetable(data_loader.dataset_df, fullfile(result_path, 'dataset_valid.csv'));