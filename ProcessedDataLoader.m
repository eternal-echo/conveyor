classdef ProcessedDataLoader
    properties
        dataset_df
    end
    
    methods
        function obj = ProcessedDataLoader(file_path)
            obj.dataset_df = readtable(file_path);
        end
        
        function workpiece_ids = getWorkpieceIDs(obj)
            workpiece_ids = unique(obj.dataset_df.WorkpieceID)';
        end
        
        function measurement_ids = getMeasurementIDs(obj, workpiece_id)
            measurement_ids = unique(obj.dataset_df.MeasurementID(obj.dataset_df.WorkpieceID == workpiece_id))';
        end
        
        % 获取dataset_df中的所有数据
        function dataset = getDataset(obj)
            dataset = obj.dataset_df;
        end

        % 获取工件workpiece_id的测量measurement_id的波形数据
        function result = getMeasurementData(obj, workpiece_id, measurement_id, varargin)
            % Initialize optional parameters
            p = inputParser;
            addOptional(p, 'valid_id', []);
            addOptional(p, 'time_range', []);
            parse(p, varargin{:});
            valid_id = p.Results.valid_id;
            time_range = p.Results.time_range;
            
            % Base mask for workpiece and measurement
            mask = (obj.dataset_df.WorkpieceID == workpiece_id) & ...
                   (obj.dataset_df.MeasurementID == measurement_id);
            
            % Apply additional filters if provided
            if ~isempty(valid_id)
                mask = mask & (obj.dataset_df.ValidCount == valid_id);
            end
            if ~isempty(time_range)
                mask = mask & (obj.dataset_df.RelativeTime >= time_range(1)) & ...
                             (obj.dataset_df.RelativeTime <= time_range(2));
            end
            
            % Return the filtered data
            result = obj.dataset_df(mask, :);
        end
        
        % 获取工件workpiece_id的测量measurement_id的波形数据中的有效区间
        function intervals = getValidIntervals(obj, workpiece_id, measurement_id)
            mask = (obj.dataset_df.WorkpieceID == workpiece_id) & ...
                   (obj.dataset_df.MeasurementID == measurement_id) & ...
                   (obj.dataset_df.ValidCount >= 0);
            valid_data = obj.dataset_df(mask, :);
            valid_ids = unique(valid_data.ValidCount)';
            intervals = zeros(length(valid_ids), 2);
            for i = 1:length(valid_ids)
                id = valid_ids(i);
                intervals(i, :) = [min(valid_data.RelativeTime(valid_data.ValidCount == id)), ...
                                   max(valid_data.RelativeTime(valid_data.ValidCount == id))];
            end
        end
    end
end
