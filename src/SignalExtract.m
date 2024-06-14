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
            parse(p, varargin{:});
            windowSize = p.Results.windowSize;
            hopSize = p.Results.hopSize;

            energy = SignalExtract.computeShortTimeEnergy(signal, windowSize, hopSize);
            autocorr = SignalExtract.computeShortTimeAutocorrelation(signal, windowSize, hopSize);
            autocorr1D = sum(abs(autocorr), 1);

            % Normalize features
            energy_normalized = SignalExtract.normalizeFeature(energy);
            autocorr1D_normalized = SignalExtract.normalizeFeature(autocorr1D);

            % Compute combined feature
            combined_feature = 0.5 * energy_normalized + 0.5 * autocorr1D_normalized;

        end

        %% 3. 对特征进行峰值检测
        function [peaks, locs, locs_idx, intervals] = detectFeaturesPeaks(combined_feature, time, varargin)
            % Initialize optional parameters
            p = inputParser;
            addOptional(p, 'peakBoundariesRatio', 0.3);
            addOptional(p, 'closeKernelSize', 240);
            parse(p, varargin{:});
            height_threshold = p.Results.peakBoundariesRatio;
            closeKernelSize = p.Results.closeKernelSize;

            % 闭运算
            se = strel('disk', closeKernelSize);
            combined_feature = imclose(combined_feature, se);
            
            [peaks, locs] = findpeaks(combined_feature, time, 'MinPeakHeight', 0.1, 'MinPeakDistance', 1, 'MinPeakProminence', 0.05, 'MinPeakWidth', 0.01);
            [~, locs_idx] = ismember(locs, time);
            intervals = SignalExtract.getPeakIntervals(locs_idx, combined_feature, time, height_threshold);
            % [intervals, peaks, locs, locs_idx] = SignalExtract.processIntervals(intervals, peaks, locs, locs_idx);
        end

        %% 4. 有效区间检测
        % Signal Envelope Extraction
        function envelope_signal = extractEnvelope(signal)
            abs_signal = abs(signal);
            [envelope_signal, ~] = envelope(abs_signal, 50, 'rms');
            se = strel('disk', 120);
            envelope_signal = imclose(envelope_signal, se);
            envelope_signal = SignalExtract.normalizeFeature(envelope_signal);
        end

        %  利用已经提取的特征峰值区间，重新寻找更准确的包络线峰值区间
        function [valid_peaks_idx, valid_intervals] = detectValidIntervals(envelope_signal, time, peak_indices, peak_intervals, varargin)
            % Initialize optional parameters
            p = inputParser;
            addOptional(p, 'threshold', 0.3); % 可选参数：阈值
            parse(p, varargin{:});
            threshold = p.Results.threshold;

            % 检查peak_intervals和peak_indices是否大小一致
            if size(peak_intervals, 2) ~= length(peak_indices)
                error('The number of peak intervals and peak indices should be the same.');
            end

            valid_peaks_idx = zeros(1, length(peak_indices));
            valid_intervals = zeros(2, length(peak_indices));
            
            search_range = 50;
            grad = gradient(envelope_signal);
            for i = 1:length(peak_indices)
                peak_interval = peak_intervals(:, i);

                % 找到当前波峰区域的最大值索引为peak_idx
                time_mask = time >= peak_interval(1) & time <= peak_interval(2);
                [~, max_idx] = max(envelope_signal(time_mask));
                peak_idx = find(time_mask, 1, 'first') + max_idx - 1;
                valid_peaks_idx(i) = peak_idx;

                left_trough = SignalExtract.findLeftTrough(envelope_signal, grad, peak_idx, threshold, search_range);
                right_trough = SignalExtract.findRightTrough(envelope_signal, grad, peak_idx, threshold, search_range);
                
                valid_intervals(:, i) = [time(left_trough); time(right_trough)];
            end
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
    end
    methods (Static, Access = private)
    
        % Helper function to find the left trough
        function left_trough = findLeftTrough(envelope_signal, grad, peak_idx, threshold, search_range)
            left_trough = peak_idx;
            % 当梯度全小于等于0（左侧拐点）且波谷小于峰值的0.3
            while left_trough > search_range
                if (all(grad(left_trough-search_range:left_trough) <= 0) && envelope_signal(left_trough) < threshold * envelope_signal(peak_idx))
                    break;
                end
                left_trough = left_trough - 1;
            end
            left_trough_last = left_trough;
            % 继续遍历到不满足上述条件
            while left_trough > search_range && (all(grad(left_trough-search_range:left_trough) == 0)) 
                left_trough = left_trough - 1;
            end
            left_trough = floor((left_trough + left_trough_last) / 2);
            if left_trough >= search_range
                left_trough = left_trough - search_range + 1;
            end
        end

        % Helper function to find the right trough
        function right_trough = findRightTrough(envelope_signal, grad, peak_idx, threshold, search_range)
            right_trough = peak_idx;
            while right_trough < length(envelope_signal) - search_range
                if (all(grad(right_trough:right_trough+search_range) >= 0) && envelope_signal(right_trough) < threshold * envelope_signal(peak_idx))
                    break;
                end
                right_trough = right_trough + 1;
            end
            right_trough_last = right_trough;
            while right_trough < length(envelope_signal) - search_range && (all(grad(right_trough:right_trough+search_range) == 0))
                right_trough = right_trough + 1;
            end
            right_trough = floor((right_trough + right_trough_last) / 2);
            if right_trough <= length(envelope_signal) - search_range
                right_trough = right_trough + search_range - 1;
            end
        end

        % Function to normalize a feature
        function normalized_feature = normalizeFeature(feature)
            normalized_feature = (feature - min(feature)) / (max(feature) - min(feature));
        end

        % Function to get peak intervals
        function peak_intervals = getPeakIntervals(locs_idx, signal, time, height_threshold)
            % left_ips = zeros(1, length(locs_idx));
            % right_ips = zeros(1, length(locs_idx));
            % for i = 1:length(locs_idx)
            %     left_base = locs_idx(i);
            %     while left_base > 1 && signal(left_base) > height_threshold * peaks(i)
            %         left_base = left_base - 1;
            %     end
            %     left_ips(i) = time(left_base);

            %     right_base = locs_idx(i);
            %     while right_base < length(time) && signal(right_base) > height_threshold * peaks(i)
            %         right_base = right_base + 1;
            %     end
            %     right_ips(i) = time(right_base);
            % end
            % peak_intervals = [left_ips; right_ips];

            grad = gradient(signal);
            peak_intervals = zeros(2, length(locs_idx));
            for i = 1:length(locs_idx)
                left_trough = SignalExtract.findLeftTrough(signal, grad, locs_idx(i), height_threshold, 50);
                right_trough = SignalExtract.findRightTrough(signal, grad, locs_idx(i), height_threshold, 50);
                peak_intervals(:, i) = [time(left_trough); time(right_trough)];
            end
        end
        

        % function [updated_intervals, updated_peaks, updated_locs, updated_locs_idx] = processIntervals(peak_intervals, peaks, locs, locs_idx)
        %     min_distance = 2; % 最小间距2秒
        %     num_peaks = length(peaks);
            
        %     % 初始化输出
        %     updated_intervals = [];
        %     updated_peaks = [];
        %     updated_locs = [];
        %     updated_locs_idx = [];
            
        %     % 按峰值从大到小排序
        %     [sorted_peaks, sort_idx] = sort(peaks, 'descend');
        %     sorted_intervals = peak_intervals(:, sort_idx);
        %     sorted_locs = locs(sort_idx);
        %     sorted_locs_idx = locs_idx(sort_idx);
            
        %     % 用于存储已经选择的峰值时间位置
        %     selected_locs = [];
            
        %     for i = 1:num_peaks
        %         current_interval = sorted_intervals(:, i);
        %         current_peak = sorted_peaks(i);
        %         current_loc = sorted_locs(i);
        %         current_loc_idx = sorted_locs_idx(i);
                
        %         % 检查当前峰值时间位置与已选择的峰值时间位置的差距是否小于2秒
        %         is_too_close = any(abs(selected_locs - current_loc) < min_distance);
                
        %         % 检查当前区间是否与已选择的区间重叠
        %         is_overlapping = false;
        %         for j = 1:size(updated_intervals, 2)
        %             if (current_interval(1) <= updated_intervals(2, j) && current_interval(2) >= updated_intervals(1, j))
        %                 is_overlapping = true;
        %                 break;
        %             end
        %         end
                
        %         % 如果不太近且不重叠，则加入输出列表
        %         if ~is_too_close && ~is_overlapping
        %             updated_intervals = [updated_intervals, current_interval];
        %             updated_peaks = [updated_peaks, current_peak];
        %             updated_locs = [updated_locs, current_loc];
        %             updated_locs_idx = [updated_locs_idx, current_loc_idx];
        %             selected_locs = [selected_locs, current_loc];
        %         end
        %     end
        % end
        

        % % 去除重叠的波峰区间
        % function [updated_intervals, updated_peaks, updated_locs, updated_locs_idx] = processIntervals(peak_intervals, peaks, locs, locs_idx)
        %     updated_intervals = zeros(2, 0); % 初始化为2xN矩阵
        %     updated_peaks = [];
        %     updated_locs = [];
        %     updated_locs_idx = [];
            
        %     if ~isempty(peak_intervals)
        %         peak_intervals = sortrows(peak_intervals')'; % 确保按起始时间排序
        %         current_interval = peak_intervals(:, 1);
        %         current_peaks = peaks(1);
        %         current_locs = locs(1);
        %         current_locs_idx = locs_idx(1);
                
        %         for i = 2:size(peak_intervals, 2)
        %             if peak_intervals(1, i) <= current_interval(2)
        %                 current_interval(2) = max(current_interval(2), peak_intervals(2, i));
        %                 current_peaks = [current_peaks, peaks(i)];
        %                 current_locs = [current_locs, locs(i)];
        %                 current_locs_idx = [current_locs_idx, locs_idx(i)];
        %             else
        %                 updated_intervals = [updated_intervals, current_interval];
        %                 updated_peaks = [updated_peaks, max(current_peaks)];
        %                 updated_locs = [updated_locs, mean(current_locs)];
        %                 updated_locs_idx = [updated_locs_idx, round(mean(current_locs_idx))];
                        
        %                 current_interval = peak_intervals(:, i);
        %                 current_peaks = peaks(i);
        %                 current_locs = locs(i);
        %                 current_locs_idx = locs_idx(i);
        %             end
        %         end
                
        %         updated_intervals = [updated_intervals, current_interval];
        %         updated_peaks = [updated_peaks, max(current_peaks)];
        %         updated_locs = [updated_locs, mean(current_locs)];
        %         updated_locs_idx = [updated_locs_idx, round(mean(current_locs_idx))];
        %     end
        % end

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
