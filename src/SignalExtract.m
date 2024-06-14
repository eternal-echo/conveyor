classdef SignalExtract
    properties
        % 可以添加属性，根据需要进行定义
    end
    
    methods (Static)
        %% 1. 去噪和基线漂移校正
        function denoised_signal = denoiseSignal(signal, time)
            denoised_signal = wdenoise(signal, 10, ...
                'Wavelet', 'sym2', ...
                'DenoisingMethod', 'BlockJS', ...
                'ThresholdRule', 'James-Stein', ...
                'NoiseEstimate', 'LevelIndependent');
            denoised_signal = detrend(denoised_signal);
        end

        %% 2. 提取特征
        function [energy, autocorr, autocorr1D, combined_feature] = extractFeatures(signal, time, varargin)
            % Initialize optional parameters
            p = inputParser;
            addOptional(p, 'windowSize', 40);
            addOptional(p, 'hopSize', 1);
            addOptional(p, 'closeKernelSize', 240);
            parse(p, varargin{:});
            windowSize = p.Results.windowSize;
            hopSize = p.Results.hopSize;
            closeKernelSize = p.Results.closeKernelSize;

            energy = SignalExtract.computeShortTimeEnergy(signal, windowSize, hopSize);
            autocorr = SignalExtract.computeShortTimeAutocorrelation(signal, windowSize, hopSize);
            autocorr1D = sum(abs(autocorr), 1);

            % Normalize features
            energy_normalized = SignalExtract.normalizeFeature(energy);
            autocorr1D_normalized = SignalExtract.normalizeFeature(autocorr1D);

            % Compute combined feature
            combined_feature = 0.5 * energy_normalized + 0.5 * autocorr1D_normalized;

            % % 最大值滤波
            % window_size = 240;
            % combined_feature = movmax(combined_feature, window_size);

            % % 最小值滤波
            % combined_feature = movmin(combined_feature, window_size);

            % 闭运算
            se = strel('disk', closeKernelSize);
            combined_feature = imclose(combined_feature, se);
        end

        %% 3. 对特征进行峰值检测和间隔合并
        function [peaks, locs, locs_idx, merged_intervals] = detectFeaturesPeaks(combined_feature, time, varargin)
            % Initialize optional parameters
            p = inputParser;
            addOptional(p, 'peakBoundariesRatio', 0.3);
            parse(p, varargin{:});
            height_threshold = p.Results.peakBoundariesRatio;
            [peaks, locs] = findpeaks(combined_feature, time, 'MinPeakHeight', 0.05, 'MinPeakDistance', 0.005, 'MinPeakProminence', 0.05, 'MinPeakWidth', 0.01);
            [~, locs_idx] = ismember(locs, time);
            peak_intervals = SignalExtract.getPeakIntervals(locs_idx, peaks, combined_feature, time, height_threshold);
            [merged_intervals, peaks, locs, locs_idx] = SignalExtract.mergeIntervals(peak_intervals, peaks, locs, locs_idx);
        end

        % Function to plot denoised signal with peak intervals
        function plotSignalWithIntervals(time, denoised_signal, intervals, varargin)
            p = inputParser;
            addOptional(p, 'title', '信号及其波峰区间');
            addOptional(p, 'markers', []); % 可选参数：标记点，格式为 [时间; 值]
            parse(p, varargin{:});
            title_str = p.Results.title;
            markers = p.Results.markers;

            figure;
            plot(time, denoised_signal);
            title(title_str);
            xlabel('时间 (秒)');
            ylabel('幅度');
            grid on;
            hold on;
            for i = 1:size(intervals, 2)
                fill([intervals(1, i), intervals(2, i), intervals(2, i), intervals(1, i)], ...
                    [min(denoised_signal), min(denoised_signal), max(denoised_signal), max(denoised_signal)], ...
                    'y', 'FaceAlpha', 0.3);
            end

            % 绘制标记点
            if ~isempty(markers)
                for j = 1:size(markers, 2)
                    plot(markers(1, j), markers(2, j), 'rx', 'MarkerSize', 8, 'LineWidth', 1);
                end
            end
            hold off;
        end


        %% 4. 有效区间检测
        % Signal Envelope Extraction
        function envelope_signal = extractEnvelope(signal)
            abs_signal = abs(signal);
            [envelope_signal, ~] = envelope(abs_signal, 50, 'rms');
            se = strel('disk', 80);
            envelope_signal = imclose(envelope_signal, se);
            envelope_signal = SignalExtract.normalizeFeature(envelope_signal);
        end

        % Valid Interval Detection
        function [valid_peaks_idx, valid_intervals] = detectValidIntervals(envelope_signal, time, peak_indices, peak_intervals, varargin)
            % Initialize optional parameters
            p = inputParser;
            addOptional(p, 'threshold', 0.3); % 可选参数：阈值
            parse(p, varargin{:});
            threshold = p.Results.threshold;

            valid_intervals = zeros(2, length(peak_indices));
            valid_peaks_idx = zeros(1, length(peak_indices));
            search_range = 50;
            grad = gradient(envelope_signal);
            
            % 检查peak_intervals和peak_indices是否大小一致
            if size(peak_intervals, 2) ~= length(peak_indices)
                error('The number of peak intervals and peak indices should be the same.');
            end
            for i = 1:length(peak_indices)
                peak_idx = peak_indices(i);
                peak_interval = peak_intervals(:, i);

                % 找到当前波峰区域的最大值索引为peak_idx
                time_mask = time >= peak_interval(1) & time <= peak_interval(2);
                [~, max_idx] = max(envelope_signal(time_mask));
                peak_idx = find(time_mask, 1, 'first') + max_idx - 1;
                valid_peaks_idx(i) = peak_idx;

                % Find the left trough
                left_trough = peak_idx;
                while left_trough > search_range
                    % Check if the gradient is zero and the envelope signal is below a threshold
                    if (all(grad(left_trough-search_range:left_trough) == 0) && envelope_signal(left_trough) < threshold * envelope_signal(peak_idx))
                        break;
                    end
                    left_trough = left_trough - 1;
                end
                if left_trough == search_range
                    left_trough = 1;
                end
                
                % Find the right trough
                right_trough = peak_idx;
                while right_trough < length(envelope_signal) - search_range
                    % Check if the gradient is zero and the envelope signal is below a threshold
                    if (all(grad(right_trough:right_trough+search_range) == 0) && envelope_signal(right_trough) < threshold * envelope_signal(peak_idx))
                        break;
                    end
                    right_trough = right_trough + 1;
                end
                if right_trough == length(envelope_signal) - search_range
                    right_trough = length(envelope_signal);
                end
                
                valid_intervals(:, i) = [time(left_trough); time(right_trough)];
            end
        end
    end

    methods (Static, Access = private)
        % Function to normalize a feature
        function normalized_feature = normalizeFeature(feature)
            normalized_feature = (feature - min(feature)) / (max(feature) - min(feature));
        end

        % Function to get peak intervals
        function peak_intervals = getPeakIntervals(locs_idx, peaks, combined_feature, time, height_threshold)
            left_ips = zeros(1, length(locs_idx));
            right_ips = zeros(1, length(locs_idx));
            for i = 1:length(locs_idx)
                left_base = locs_idx(i);
                while left_base > 1 && combined_feature(left_base) > height_threshold * peaks(i)
                    left_base = left_base - 1;
                end
                left_ips(i) = time(left_base);

                right_base = locs_idx(i);
                while right_base < length(time) && combined_feature(right_base) > height_threshold * peaks(i)
                    right_base = right_base + 1;
                end
                right_ips(i) = time(right_base);
            end
            peak_intervals = [left_ips; right_ips];
        end

        % Function to merge intervals and update peaks, locs, locs_idx
        function [merged_intervals, updated_peaks, updated_locs, updated_locs_idx] = mergeIntervals(peak_intervals, peaks, locs, locs_idx)
            merged_intervals = zeros(2, 0); % 初始化为2xN矩阵
            updated_peaks = [];
            updated_locs = [];
            updated_locs_idx = [];
            
            if ~isempty(peak_intervals)
                peak_intervals = sortrows(peak_intervals')'; % 确保按起始时间排序
                current_interval = peak_intervals(:, 1);
                current_peaks = peaks(1);
                current_locs = locs(1);
                current_locs_idx = locs_idx(1);
                
                for i = 2:size(peak_intervals, 2)
                    if peak_intervals(1, i) <= current_interval(2)
                        current_interval(2) = max(current_interval(2), peak_intervals(2, i));
                        current_peaks = [current_peaks, peaks(i)];
                        current_locs = [current_locs, locs(i)];
                        current_locs_idx = [current_locs_idx, locs_idx(i)];
                    else
                        merged_intervals = [merged_intervals, current_interval];
                        updated_peaks = [updated_peaks, max(current_peaks)];
                        updated_locs = [updated_locs, mean(current_locs)];
                        updated_locs_idx = [updated_locs_idx, round(mean(current_locs_idx))];
                        
                        current_interval = peak_intervals(:, i);
                        current_peaks = peaks(i);
                        current_locs = locs(i);
                        current_locs_idx = locs_idx(i);
                    end
                end
                
                merged_intervals = [merged_intervals, current_interval];
                updated_peaks = [updated_peaks, max(current_peaks)];
                updated_locs = [updated_locs, mean(current_locs)];
                updated_locs_idx = [updated_locs_idx, round(mean(current_locs_idx))];
            end
        end

        % Function to compute short-time energy
        function energy = computeShortTimeEnergy(signal, windowSize, hopSize)
            if size(signal, 1) < size(signal, 2)
                signal = signal';
            end
            signal = signal / max(abs(signal));
            frames = buffer(signal, windowSize, windowSize - hopSize, 'nodelay');
            energy = sum(frames.^2, 1);
            energy = interp1(1:length(energy), energy, linspace(1, length(energy), length(signal)), 'linear', 'extrap');
        end

        % Function to compute short-time autocorrelation
        function autocorr = computeShortTimeAutocorrelation(signal, windowSize, hopSize)
            if size(signal, 1) < size(signal, 2)
                signal = signal';
            end
            signal = signal / max(abs(signal));
            frames = buffer(signal, windowSize, windowSize - hopSize, 'nodelay');
            numFrames = size(frames, 2);
            autocorr = zeros(windowSize, numFrames);
            for i = 1:numFrames
                ac = xcorr(frames(:, i), 'biased');
                autocorr(:, i) = ac(windowSize:end);
            end
            interp_autocorr = zeros(windowSize, length(signal));
            for i = 1:windowSize
                interp_autocorr(i, :) = interp1(1:numFrames, autocorr(i, :), linspace(1, numFrames, length(signal)), 'linear', 'extrap');
            end
            autocorr = interp_autocorr;
        end
    end
end
