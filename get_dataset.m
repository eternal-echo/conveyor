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

% 示例：获取第一个工件的第一个测量数据
disp('第一个工件的第一个测量数据：');
measurement_data = workpiece_dataset.get_single_measurement(1, 0);
disp(measurement_data);
