%% Hyperparameters
% Global parameters used in preprocessing

% Cell array of prefixes (equivalent to Python list)
PREFIXES = {"test_data/neutral/neutral_", "test_data/forced/forced_"};

% Cell array of filenames (equivalent to Python list)
FILENAMES = {"angry", "disgusted", "happy", "surprised"};

% Number of channels (scalar)
NUM_CHANNELS = 3;

% Unwanted channels (empty array)
UNWANTED_CHANNELS = [];  

% Spectrogram and signal processing parameters
NFFT = 128;            % Number of FFT points
SAMPLE_RATE = 1152;    % Sample rate in Hz
LOW_CUTOFF = 50;       % Low cutoff frequency in Hz
HIGH_CUTOFF = 200;     % High cutoff frequency in Hz
%% Loading data
%-------------------------------------------------------------

% Initialize empty cell array for raw_data
raw_data = cell(1, length(FILENAMES));

% Loop through the filenames (emotions)
for file_id = 1:length(FILENAMES)
    a = cell(1, length(PREFIXES));  % Temporary storage for the current emotion data

    % Loop through the prefixes (modalities: voiced/silent)
    for prefix_id = 1:length(PREFIXES)
        % Construct the full file path
        file_path = strcat(PREFIXES{prefix_id}, FILENAMES{file_id}, '.txt');
        
        % Open the file and read the contents
        fid = fopen(file_path, 'r');
        if fid == -1
            error('Failed to open file: %s', file_path);
        end

        % Read the file into a cell array of lines
        temp = textscan(fid, '%s', 'Delimiter', '\n');
        temp = temp{1};  % Extract the actual data from the cell
        fclose(fid);

        % Determine number of points
        num_points = length(temp);

        % Initialize matrix to store the sensor data
        temp_data = zeros(NUM_CHANNELS, num_points);

        % Parse each line and split by commas
        for c = 1:num_points
            split_line = str2double(strsplit(temp{c}, ','));
            temp_data(:, c) = split_line(1:NUM_CHANNELS);  % Store data for each channel
        end

        % Store the length of temp_data and the data itself in the cell
        a{prefix_id} = {num_points, temp_data};
    end

    % Find the maximum number of points across all modalities
    max_points = max(cellfun(@(x) x{1}, a));

    % Resize the data matrices to ensure consistency across modalities
    for prefix_id = 1:length(a)
        temp_data = a{prefix_id}{2};  % Extract the temp_data matrix
        if size(temp_data, 2) < max_points
            temp_data(:, end+1:max_points) = 0;  % Pad with zeros if necessary
        end
        a{prefix_id}{2} = temp_data;  % Update the resized data
    end

    % Store the data for both modalities (voiced and silent) in raw_data
    raw_data{file_id} = cellfun(@(x) x{2}, a, 'UniformOutput', false);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The shape of the data format is 3-dimensional:
%   1st dimension: emotion
%   2nd dimension: modality (voiced/silent)
%   3rd dimension: sEMG channels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Preprocessing
% Normalize signals
preprocessed_data = cell(1, length(FILENAMES));
DOWNSCALE_FACTOR = 19;
% For every emotion
for emotion_index = 1:length(raw_data)
    preprocessed_data{emotion_index} = cell(1, length(raw_data{emotion_index}));
    % For every modality
    for modality_index = 1:length(raw_data{emotion_index})
        %modality_data = zeros(NUM_CHANNELS, fix(size(raw_data{emotion_index}{modality_index}, 2) / DOWNSCALE_FACTOR) + 1);
        modality_data = zeros(NUM_CHANNELS, fix(size(raw_data{emotion_index}{modality_index}, 2)));
        % For every channel
        for channel_index = 1:NUM_CHANNELS
            k = raw_data{emotion_index}{modality_index}(channel_index, :);  % Access channel data
            %modality_data(channel_index, :) = k;
            % Process the data for each channel
            %normalized_and_bandpass = normalize(bandpass(downsample(k, DOWNSCALE_FACTOR), [LOW_CUTOFF HIGH_CUTOFF], SAMPLE_RATE / DOWNSCALE_FACTOR), 'range', [-1 1]);
            %modality_data(channel_index, :) = smoothdata(normalized_and_bandpass, "movmedian",5);
            %modality_data(channel_index, :) = bandpass(normalize(k, 'range', [0 1]), [LOW_CUTOFF HIGH_CUTOFF], SAMPLE_RATE);
            modality_data(channel_index, :) = smoothdata(k, "movmean",5);
        end
        preprocessed_data{emotion_index}{modality_index} = modality_data;
    end
end
%% Plotting - All
%-------------------------------------------------------------
plot_data = preprocessed_data;
MODALITY = 1;  % In MATLAB, indexing starts at 1

% Create a figure with subplots for each emotion and channel
figure;
sgtitle(sprintf('Comparing expressions in modality %s', PREFIXES{MODALITY}));

% Loop through each emotion (row) and channel (columns)
for c = 1:length(FILENAMES)
    % Set ylabel to the emotion
    for d = 1:NUM_CHANNELS
        % Create subplot for current emotion and channel
        subplot(length(FILENAMES), NUM_CHANNELS, (c-1)*NUM_CHANNELS + d);
        
        % Extract data for the current channel
        j = plot_data{c}{MODALITY}(d, :);
        
        % Create a time axis
        time_axis = linspace(0, length(j) / SAMPLE_RATE, length(j));
        
        % Plot the data
        plot(time_axis, j, '-');
        
        % Set y-label for the first column
        if d == 1
            ylabel(FILENAMES{c});
        end
        
        % Set x-ticks to major intervals
        xticks(0:10:max(time_axis));
        
        % Only label outer ticks
        if c < length(FILENAMES)
            set(gca, 'XTickLabel', []);
        end
    end
end

% Add common xlabel (since MATLAB doesn't have a direct "supxlabel" equivalent)
xlabel('Time (seconds)', 'Position', [0.5, -0.05]);

% Add common labels for each channel (0, 1, 2)
annotation('textbox', [0.15, 0.9, 0.1, 0.1], 'String', 'Channel 0', 'EdgeColor', 'none');
annotation('textbox', [0.45, 0.9, 0.1, 0.1], 'String', 'Channel 1', 'EdgeColor', 'none');
annotation('textbox', [0.75, 0.9, 0.1, 0.1], 'String', 'Channel 2', 'EdgeColor', 'none');

% Show the plot
hold off;
%%
plot_data = raw_data;
MODALITY = 1;  % In MATLAB, indexing starts at 1

% Create a figure with subplots for each emotion and channel
figure;
sgtitle(sprintf('Comparing expressions in modality %s', PREFIXES{MODALITY}));

% Loop through each emotion (row) and channel (columns)
for c = 1:length(FILENAMES)
    % Set ylabel to the emotion
    for d = 1:NUM_CHANNELS
        % Create subplot for current emotion and channel
        subplot(length(FILENAMES), NUM_CHANNELS, (c-1)*NUM_CHANNELS + d);
        
        % Extract data for the current channel
        j = plot_data{c}{MODALITY}(d, :);
        
        % Create a time axis
        time_axis = linspace(0, length(j) / SAMPLE_RATE, length(j));
        
        % Plot the data
        plot(time_axis, j, '-');
        
        % Set y-label for the first column
        if d == 1
            ylabel(FILENAMES{c});
        end
        
        % Set x-ticks to major intervals
        xticks(0:10:max(time_axis));
        
        % Only label outer ticks
        if c < length(FILENAMES)
            set(gca, 'XTickLabel', []);
        end
    end
end
%% Plotting - for testing
%-------------------------------------------------------------
plot_data = cellfun(@(x) x{1}(2, :), preprocessed_data, 'UniformOutput', false);

figure;
num_plots = length(plot_data);  % Number of subplots

for i = 1:num_plots
    subplot(2, 2, i);  % Create a 2x2 grid of subplots, plotting on the i-th subplot
    plot(plot_data{i}, '-o');  % Plot the i-th vector
    title([num2str(FILENAMES{i})]);  % Add a title to each subplot
end
xlabel('Time');
sgtitle('Comparing cheek (channel 0) EMG signal');

% bandpass, downsample,
% normalize (by range, not value registered)/smoothdata(), envelope
% 
% look at computing resources, data needed, try setting some up
%
% ask MyoWare about sensor output