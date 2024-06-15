addpath('src');
addpath('data');
data_path = 'data';
result_path = 'results';
feature_path = fullfile(result_path, 'features');
visualization_path = fullfile(result_path, 'features', 'visualization');

% 创建保存结果的文件夹
if ~exist(feature_path, 'dir')
    mkdir(feature_path);
end
if ~exist(visualization_path, 'dir')
    mkdir(visualization_path);
end

% 加载数据集
data_loader = DataLoader(fullfile(result_path, 'dataset_valid.csv'));

% 初始化全局特征表
all_features = [];
feature_info = [];
window_labels = [];

% 窗口参数
window_size = 120; % 窗口大小
step_size = 20; % 窗口步长

% 遍历每个工件和测量
workpiece_ids = data_loader.getWorkpieceIDs();
for i = 1:length(workpiece_ids)
    workpiece_id = workpiece_ids(i);
    measurement_ids = data_loader.getMeasurementIDs(workpiece_id);

    for j = 1:length(measurement_ids)
        measurement_id = measurement_ids(j);
        data = data_loader.getMeasurementData(workpiece_id, measurement_id);
        time = data.RelativeTime;
        signal = data.CH1;

        % 获取有效区间
        valid_intervals = data_loader.getValidIntervals(workpiece_id, measurement_id);

        % 可视化信号和有效区间
        figure('Visible', 'off');
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
            patch([start_time, end_time, end_time, start_time], ...
                [min(signal), min(signal), max(signal), max(signal)], ...
                [0.9, 0.9, 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        end
        hold off;

        % 保存图像到文件
        image_file = fullfile(visualization_path, sprintf('signal%d_%d.png', workpiece_id, measurement_id));
        saveas(gcf, image_file);
        close(gcf);

        % 滑动窗口遍历信号
        fs = 1.0 / (time(2) - time(1));
        features = [];
        labels = [];
        feature_labels = {'Energy', 'Amplitude', 'Max', 'Spectral Centroid', 'Spectral Peak', 'Autocorrelation Peak'};
        num_windows = floor((length(signal) - window_size) / step_size) + 1;

        for k = 1:num_windows
            start_idx = (k-1) * step_size + 1;
            end_idx = start_idx + window_size - 1;
            if end_idx > length(signal)
                break;
            end
            % 提取窗口部分的信号
            segment = signal(start_idx:end_idx);
            segment_time = time(start_idx:end_idx);
            
            % 确定窗口是否包含有效信号（如果有效信号的持续时间超过窗口大小的一半，则将窗口标记为有效。）
            valid_duration = sum(arrayfun(@(start_time, end_time) sum(segment_time >= start_time & segment_time <= end_time), valid_intervals(1,:), valid_intervals(2,:)));
            is_valid = (valid_duration / window_size) > 0.5;
            labels = [labels; is_valid];

            % 计算特征
            energy = sum(segment.^2);
            amplitude = mean(abs(segment));
            max_val = max(segment);
            fft_vals = abs(fft(segment));
            fft_vals = fft_vals(1:ceil(length(segment)/2));
            freqs = linspace(0, fs/2, length(fft_vals))';
            spectral_centroid = sum(freqs .* fft_vals) / sum(fft_vals);
            [~, max_index] = max(fft_vals);
            spectral_peak = freqs(max_index);
            autocorr_value = xcorr(segment, 'coeff');
            autocorr_peak = max(autocorr_value);
            
            % 汇总特征
            features = [features; energy, amplitude, max_val, spectral_centroid, spectral_peak, autocorr_peak];
        end

        % 汇总所有特征数据
        all_features = [all_features; features];
        feature_info = [feature_info; repmat([workpiece_id, measurement_id], size(features, 1), 1)];
        window_labels = [window_labels; labels];

        % 可视化并保存每个特征的图像
        for feature_index = 1:length(feature_labels)
            figure('Visible', 'off');
            % 画出所有窗口的特征值
            plot(features(:, feature_index), 'o-', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7]);

            hold on;
            % 突出显示有效窗口
            valid_indices = find(labels); % 找出有效窗口的索引
            plot(valid_indices, features(valid_indices, feature_index), 'ro', 'MarkerFaceColor', 'r'); % 使用红色标记有效窗口

            title([feature_labels{feature_index}, ' for Workpiece ', num2str(workpiece_id), ', Measurement ', num2str(measurement_id)]);
            xlabel('Window Index');
            ylabel(feature_labels{feature_index});
            
            feature_visualization_file = fullfile(visualization_path, sprintf('%s_%d_%d.png', feature_labels{feature_index}, workpiece_id, measurement_id));
            saveas(gcf, feature_visualization_file);
            close(gcf);
        end
    end
end

% 保存特征到一个CSV文件
feature_file = fullfile(feature_path, 'all_features.csv');
header = {'WorkpieceID', 'MeasurementID', 'Energy', 'Amplitude', 'Max', 'Spectral Centroid', 'Spectral Peak', 'Autocorrelation Peak', 'Label'};
all_feature_table = array2table([feature_info, all_features, window_labels], 'VariableNames', header);
writetable(all_feature_table, feature_file);

% PCA分析和可视化
% 读取特征数据
data = table2array(all_feature_table(:, 3:end-1));
[coeff, score, ~, ~, explained] = pca(data);

% 散点图
figure;
gscatter(score(:,1), score(:,2), all_feature_table.Label);
title('PCA Scatter Plot');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
grid on;
saveas(gcf, fullfile(visualization_path, 'pca_scatter.png'));
close(gcf);

% 热力图
figure;
imagesc(corr(data));
colorbar;
title('Feature Correlation Heatmap');
xlabel('Features');
ylabel('Features');
set(gca, 'XTick', 1:length(feature_labels), 'XTickLabel', feature_labels, 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:length(feature_labels), 'YTickLabel', feature_labels);
saveas(gcf, fullfile(visualization_path, 'correlation_heatmap.png'));
close(gcf);
