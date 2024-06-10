classdef WorkpieceDataset
    properties
        dataset_df
        workpiece_ids
        measurements
        workpiece_count
    end
    
    methods
        function obj = WorkpieceDataset(dataset_df)
            % 初始化 WorkpieceDataset 类
            obj.dataset_df = dataset_df;
            obj.workpiece_ids = unique(dataset_df.WorkpieceID)';
            obj.measurements = zeros(size(obj.workpiece_ids));
            
            for i = 1:length(obj.workpiece_ids)
                workpiece_id = obj.workpiece_ids(i);
                % 查询每个工件的最大测量次数
                max_measurement = max(dataset_df.MeasurementID(dataset_df.WorkpieceID == workpiece_id)) + 1;
                obj.measurements(i) = max_measurement;
            end
            obj.workpiece_count = length(obj.workpiece_ids);
        end
        
        function result = get_single_measurement(obj, workpiece_id, measurement_id, start_time, end_time)
            % 获取特定工件的单次测量数据
            if nargin < 4
                start_time = 0;
            end
            if nargin < 5
                end_time = max(obj.dataset_df.RelativeTime);
            end
            mask = (obj.dataset_df.WorkpieceID == workpiece_id) & ...
                   (obj.dataset_df.MeasurementID == measurement_id) & ...
                   (obj.dataset_df.RelativeTime >= start_time) & ...
                   (obj.dataset_df.RelativeTime <= end_time);
            result = obj.dataset_df(mask, :);
        end
    end
end
