%% 加载测试的数据集
% 加载数据集
dataset_df = readtable('dataset.csv');

% 示例：查看前几行数据
disp(dataset_df(1:5, :));

% 创建数据集对象
workpiece_dataset = WorkpieceDataset(dataset_df);

% 查看工件编号
disp('工件编号：');
disp(workpiece_dataset.workpiece_ids);

% 查看工件数量
disp('工件数量：');
disp(workpiece_dataset.workpiece_count);

% 查看每个工件的测量次数
disp('每个工件的测量次数：');
disp(workpiece_dataset.measurements);


% 获取信号和时间数据
workpiece_id = 5;
measurement_id = 1;

% 示例：获取第一个工件的第一个测量数据
disp('第一个工件的第一个测量数据：');
measurement_data = workpiece_dataset.get_single_measurement(workpiece_id, measurement_id);
disp(measurement_data);


[time, signal] = workpiece_dataset.get_time_and_signal(workpiece_id, measurement_id);

% 显示信号和时间数据
disp('时间数据：');
disp(time);
disp('信号数据：');
disp(signal);

% 绘制信号数据
figure;
plot(time, signal);
title('Signal Data');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;


save('test_data.mat', 'time', 'signal')
