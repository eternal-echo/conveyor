
load('test_data.mat');

denoised_signal = wdenoise(signal,10, ...
    Wavelet='sym2', ...
    DenoisingMethod='BlockJS', ...
    ThresholdRule='James-Stein', ...
    NoiseEstimate='LevelIndependent');

denoised_signal = detrend(denoised_signal);
    
% 绘制原始信号
figure;
subplot(3, 1, 1);
plot(time, denoised_signal);
title('去噪信号');
xlabel('时间 (秒)');
ylabel('幅度');

windowSize = 40; % 根据具体情况调整
hopSize = 1; % 每次移动的样本数，可以根据实际情况调整

energy = computeShortTimeEnergy(denoised_signal, windowSize, hopSize);

% 可视化短时能量
subplot(3, 1, 2);
plot(time, energy);
title('Short-Time Energy');
xlabel('Time (s)');
ylabel('Energy');

autocorr = computeShortTimeAutocorrelation(denoised_signal, windowSize, hopSize);

% 可视化短时自相关
subplot(3, 1, 3);
imagesc(time, 1:windowSize, autocorr);
title('Short-Time Autocorrelation');
xlabel('Time (s)');
ylabel('Lag');
colorbar;

autocorr = abs(autocorr);
autocorr1D = sum(autocorr, 1);

figure;
plot(time, autocorr1D);
title('One-Dimensional Representation of Short-Time Autocorrelation');
xlabel('Frame Index');
ylabel('Value');

% 归一化处理
energy_normalized = (energy - min(energy)) / (max(energy) - min(energy));
autocorr1D_normalized = (autocorr1D - min(autocorr1D)) / (max(autocorr1D) - min(autocorr1D));

% 计算相关系数
correlation = corr(energy_normalized', autocorr1D_normalized');
disp(['Correlation Coefficient: ', num2str(correlation)]);

% 加权求和
combined_feature = 0.5 * energy_normalized + 0.5 * autocorr1D_normalized;

% 寻找峰值
% findpeaks(combined_feature, time, 'MinPeakHeight', 0.05, 'MinPeakDistance', 0.005, 'MinPeakProminence', 0.05, 'Annotate','extents', 'WidthReference','halfheight');
[peaks, locs, widths, proms] = findpeaks(combined_feature, time, 'MinPeakHeight', 0.05, 'MinPeakDistance', 0.005, 'MinPeakProminence', 0.05);
% 获取locs对应的索引
[~, locs_idx] = ismember(locs, time);
% 计算波峰区间(height_threshold*height）
height_threshold = 0.1;
left_ips = zeros(1, length(locs_idx));
right_ips = zeros(1, length(locs_idx));
for i = 1:length(locs_idx)
    left_base = locs_idx(i);
    while left_base > 1 && combined_feature(left_base) > height_threshold * peaks(i)
        left_base = left_base - 1;
    end
    left_ips(i) = time(left_base);

    right_base = locs_idx(i);
    while right_base < length(locs_idx) && combined_feature(right_base) > height_threshold * peaks(i)
        right_base = right_base + 1;
    end
    right_ips(i) = time(right_base);
end

peak_intervals = [left_ips; right_ips]';
% 合并有交集的区间
merged_intervals = [];
if ~isempty(peak_intervals)
    % 按区间的起始位置排序
    peak_intervals = sortrows(peak_intervals);

    % 初始化第一个区间
    current_interval = peak_intervals(1, :);

    for i = 2:size(peak_intervals, 1)
        % 检查当前区间是否与下一个区间重叠或相邻
        if peak_intervals(i, 1) <= current_interval(2)
            % 合并区间
            current_interval(2) = max(current_interval(2), peak_intervals(i, 2));
        else
            % 将当前区间添加到结果中，并开始一个新的区间
            merged_intervals = [merged_intervals; current_interval];
            current_interval = peak_intervals(i, :);
        end
    end

    % 添加最后一个区间
    merged_intervals = [merged_intervals; current_interval];
end

% 绘制加权求和特征及其波峰区间
figure;
plot(time, combined_feature);
title('加权求和特征及其波峰区间');
xlabel('时间 (秒)');
ylabel('幅度');
grid on;
hold on;
% 标记波峰边界点
plot(locs, peaks, 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
% 绘制波峰区间
for i = 1:size(merged_intervals, 1)
    % 用直线连接波峰区间点，直线端点为(merged_intervals(i, 1), height_threshold * peaks(i))和(merged_intervals(i, 2), height_threshold * peaks(i))
    plot([merged_intervals(i, 1), merged_intervals(i, 2)], [height_threshold * peaks(i), height_threshold * peaks(i)], 'k', 'LineWidth', 2);
end


% 绘制去噪信号及其波峰区间
figure;
plot(time, denoised_signal);
title('去噪信号及其波峰区间');
xlabel('时间 (秒)');
ylabel('幅度');
grid on;
hold on;

% 绘制波峰区间
for i = 1:size(merged_intervals, 1)
    % 用透明的矩形表示峰值区域
    rectangle('Position', [merged_intervals(i, 1), min(denoised_signal), merged_intervals(i, 2) - merged_intervals(i, 1), max(denoised_signal) - min(denoised_signal)], 'FaceColor', 'none', 'EdgeColor', [0.8, 0.8, 0.8]);
end
function energy = computeShortTimeEnergy(signal, windowSize, hopSize)
    % computeShortTimeEnergy - Computes the short-time energy of a signal.
    %
    % Syntax: energy = computeShortTimeEnergy(signal, windowSize, hopSize)
    %
    % Inputs:
    %    signal - The input signal (1D array)
    %    windowSize - The size of the window for short-time analysis
    %    hopSize - The hop size between successive windows
    %
    % Outputs:
    %    time - The time axis corresponding to the short-time energy (1D array)
    %    energy - The computed short-time energy (1D array)
    
    % Ensure signal is a column vector
    if size(signal, 1) < size(signal, 2)
        signal = signal';
    end
    
    % Normalize the signal
    signal = signal / max(abs(signal));
    
    % Frame the signal using buffer
    frames = buffer(signal, windowSize, windowSize - hopSize, 'nodelay');
    
    % Compute the short-time energy
    energy = sum(frames.^2, 1);

    % Interpolate energy to match the original signal length
    energy = interp1(1:length(energy), energy, linspace(1, length(energy), length(signal)), 'linear', 'extrap');

end

function autocorr = computeShortTimeAutocorrelation(signal, windowSize, hopSize)
    % computeShortTimeAutocorrelation - Computes the short-time autocorrelation of a signal.
    %
    % Syntax: autocorr = computeShortTimeAutocorrelation(signal, windowSize, hopSize)
    %
    % Inputs:
    %    signal - The input signal (1D array)
    %    windowSize - The size of the window for short-time analysis
    %    hopSize - The hop size between successive windows
    %
    % Outputs:
    %    autocorr - The computed short-time autocorrelation (2D array)
    
    % Ensure signal is a column vector
    if size(signal, 1) < size(signal, 2)
        signal = signal';
    end
    
    % Normalize the signal
    signal = signal / max(abs(signal));
    
    % Frame the signal using buffer
    frames = buffer(signal, windowSize, windowSize - hopSize, 'nodelay');
    
    % Compute the short-time autocorrelation
    numFrames = size(frames, 2);
    autocorr = zeros(windowSize, numFrames);
    for i = 1:numFrames
        ac = xcorr(frames(:, i), 'biased');
        autocorr(:, i) = ac(windowSize:end);  % Only keep the positive lags
    end

    % Interpolate autocorrelation to match the original signal length
    interp_autocorr = zeros(windowSize, length(signal));
    for i = 1:windowSize
        interp_autocorr(i, :) = interp1(1:numFrames, autocorr(i, :), linspace(1, numFrames, length(signal)), 'linear', 'extrap');
    end
    autocorr = interp_autocorr;
end