%% Split data segments
EEG.etc.eeglabvers = '2023.1'; % This tracks which version of EEGLAB is being used, you may ignore it
EEG = pop_select( EEG, 'channel',{'Fp1','Fp2','AF3','AF4','Fz','F3','F4','F7','F8','FC1','FC2','FC5','FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6','Pz','P3','P4','P7','P8','PO3','PO4','PO7','PO8','Oz','O1','O2'}); 
EEG = pop_eegfiltnew(EEG, 'locutoff',0.3,'hicutoff',50,'plotfreqz',1); % Apply bandpass filter (0.3 to 50 Hz)
EEG = pop_resample( EEG, 200); % Resample the EEG data to 200 Hz

% Extract emotion elicitation phase signals (total 42s, can be divided into 7*6s / 14*3s / 42*1s)
% EEG = pop_epoch( EEG, {  '1'  }, [2  44], 'newname', ' feiyu_chueng_train resampled epochs', 'epochinfo', 'yes');

%% Extract emotion judgment phase EEG signals
EEG = pop_epoch( EEG, {  '4'  }, [1  7], 'newname', ' chueng_train resampled epochs', 'epochinfo', 'yes'); % Extract 1-7 seconds of emotion judgment phase
% EEG = pop_rmbase( EEG, [],[]); % Optional: remove baseline
% EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on'); % Optional: Run ICA to separate sources
pop_eegplot( EEG, 1, 1, 1); % Visualize the EEG data

%% Plot scalp topography
EEG = pop_chanedit(EEG, {'lookup','G:\\MATLAB2020\\eeglab\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc'}); 
figure; pop_spectopo(EEG, 1, [1000  5995], 'EEG' , 'freq', [2 6 10 22 40], 'freqrange',[0 50],'electrodes','off');
figure; pop_spectopo(EEG, 4, [1000  5995], 'EEG' , 'freq', [2 6 10 22 40], 'freqrange',[0 50],'electrodes','off');
figure; pop_spectopo(EEG, 1, [4000  6995], 'EEG' , 'freq', [2 6 10 22 40], 'freqrange',[0 50],'electrodes','off');
figure; pop_spectopo(EEG, 1, [7000  9995], 'EEG' , 'freq', [2 6 10 22 40], 'freqrange',[0 50],'electrodes','off');
figure; pop_spectopo(EEG, 1, [10000  12995], 'EEG' , 'freq', [2 6 10 22 40], 'freqrange',[0 50],'electrodes','off');
figure; pop_spectopo(EEG, 1, [13000  15995], 'EEG' , 'freq', [2 6 10 22 40], 'freqrange',[0 50],'electrodes','off');
figure; pop_spectopo(EEG, 1, [16000  18995], 'EEG' , 'freq', [2 6 10 22 40], 'freqrange',[0 50],'electrodes','off');

%% Save as .mat file
variables = EEG.data;
filename = 'D:\\2025\\experiment\\car2\\car2_judge.mat';
save(filename,'variables'); % Save EEG data as .mat file

%% Data segmentation processing
% The dimensions are (34, 8400), where the two dimensions represent (number of channels, number of samples). The sampling frequency is 200Hz.
% The second dimension is split sequentially into 600*14, so that the overall dimensions become (34, 600, 14).
% Loop through files from car1 to car25 (training set: car1 to car 16; verification set: car 17 to car 25)
for carNum = 1:25
    % Create the input file path
    inputFilename = sprintf('D:\\2025\\experiment\\car%d\\car%d.mat', carNum, carNum);
    % Load the file
    loaded_data = load(inputFilename);
    % Assuming the variable name in the file is 'variables'
    a = loaded_data.variables;
%     disp(size(a));
    % Reshape the array to (34, 600, 2) / For the judgment phase, reshape to (34, 600, 2)
    a_reshaped = reshape(a, [34, 600, 2]);
    % Use the permute function to rearrange the dimensions to (2, 34, 600)
    a_permuted = permute(a_reshaped, [3, 1, 2]);
    % Create the output file path
    outputFilename = sprintf('D:\\2025\\experiment\\car%d\\car%d.mat', carNum);
    % Save the variable to a .mat file
    save(outputFilename, 'a');
end
% Remove a poorly recognized data set in training set
%% Data concatenation (training set)
% The dimensions of the emotional stimulation phase are (210, 34, 600), 
% and the dimensions of the emotional judgment phase are (30, 34, 600)

% Define file paths from car1 to car15 dynamically using a loop
file_paths = cell(1, 15);
for carNum = 1:15
    file_paths{carNum} = sprintf('D:\\2025\\experiment\\car%d\\car%d.mat', carNum);  % Dynamically create file path for each car
end

% Initialize a cell array
loaded_data_cell = cell(1, numel(file_paths));

% Load each file one by one
for i = 1:numel(file_paths)
    loaded_data = load(file_paths{i});  % Load the data
    loaded_data_cell{i} = loaded_data.a_permuted;  % Store the loaded data in the cell array
end

% Concatenate along the first dimension, changing the dimension from (34, 600, 14) to (34*15, 600, 14)
concatenated_data = cat(1, loaded_data_cell{:});

% Display the dimensions of the concatenated data
disp('Dimensions of concatenated data:');
disp(size(concatenated_data));

variables = concatenated_data;
filename = 'D:\\2025\\experiment\\code\\train\\train.mat';
save(filename, 'variables');
