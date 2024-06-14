%% 加载测试的数据集
data_path = 'data';

% 创建数据集对象
data_loader = DataLoader(fullfile(data_path, 'dataset.csv'));

dataset_df = data_loader.getDataset();

% 查看工件编号
disp('工件编号：');
workpiece_ids = data_loader.getWorkpieceIDs();
disp(workpiece_ids);

% 查看工件数量
disp('每个工件的测量次数：');
for workpiece_id = workpiece_ids
    measurement_ids = data_loader.getMeasurementIDs(workpiece_id);
    disp(['工件 ', num2str(workpiece_id), ' 的测量次数：', num2str(length(measurement_ids))]);
end

% 获取信号和时间数据
workpiece_id = 1;
measurement_id = 0;

% 示例：获取第一个工件的第一个测量数据
disp('第一个工件的第一个测量数据：');
measurement_data = data_loader.getMeasurementData(workpiece_id, measurement_id);

time = measurement_data.RelativeTime';
signal = measurement_data.CH1';

% 显示信号和时间数据
disp('时间数据：');
disp(time(1:10));
disp('信号数据：');
disp(signal(1:10));

% 绘制信号数据
figure;
plot(time, signal);
title('Signal Data');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;


% save('test_data.mat', 'data_loader', 'time', 'signal', 'workpiece_ids', 'measurement_ids', 'workpiece_id', 'measurement_id');
save(fullfile(data_path, 'test_data.mat'), 'data_loader', 'time', 'signal', 'workpiece_ids', 'measurement_ids', 'workpiece_id', 'measurement_id');
