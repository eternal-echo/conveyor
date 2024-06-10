% 生成示例信号
fs = 1000;  % 采样频率
N = 10000;  % 样本数量
time = linspace(0, N / fs, N);

% 生成有效信号
valid_signal = sin(2 * pi * 50 * time) .* (time > 2 & time < 3);  % 2s到3s之间的有效信号

% 生成低幅值噪声
low_amplitude_noise = normrnd(0, 0.2, size(time));

% 生成高频毛刺噪声
high_amplitude_spikes = zeros(size(time));
num_spikes = 50;
spike_indices = randi(length(time), 1, num_spikes);
for i = 1:num_spikes
    idx = spike_indices(i);
    high_amplitude_spikes(idx:idx+9) = normrnd(1, 0.7, 1, 10);  % 持续10个采样点的高幅值毛刺
end

% 合成信号
signal = valid_signal + low_amplitude_noise + high_amplitude_spikes;

% 显示原始信号
figure('Position', [100, 100, 1200, 600]);
plot(time, signal, 'DisplayName', 'Noisy Signal');
hold on;
plot(time, valid_signal, '--', 'DisplayName', 'Valid Signal');
legend();
title('Original Signal with Noise');
xlabel('Time [s]');
ylabel('Amplitude');
