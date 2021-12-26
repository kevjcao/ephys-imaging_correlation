%% Notes
% import raw imaging data into "unnamed" variable, import raw ephys data
% into "unnnamed1" variable -- change this later

%% variables

prompt = {'ScanImage framerate (Hz)', 'Trigger delay (s)'};
dlgtitle = 'Experiment parameters';
dims = [1 50];
expparameters = str2double(inputdlg(prompt, dlgtitle, dims));  %(1,1) is framerate, (2,1) is delay between image start and ephys

%% process ephys data

rawephys = [unnamed1(:,1)/1000 unnamed1(:,2)];
rawdFF = [(unnamed(:,1)/expparameters(1,1)) + expparameters(2,1) unnamed(:,2)];

%% plot data

rawdata = figure(1);
hold on;
yyaxis left
plot(rawephys(:,1), rawephys(:,2));
ylabel('mV');
yyaxis right
plot(rawdFF(:,1), rawdFF(:,2));
ylabel('dF/F');
xlabel('Time (s)')
hold off;

%% replot dimensions
prompt = {'Width (pixels)', 'Height (pixels)', 'Time window (s)', 'mV scale min', 'mV scale max', 'dF/F scale min', 'dF/F scale max'};
dlgtitle = 'Figure properties';
dims = [1 50];
figparameters = str2double(inputdlg(prompt, dlgtitle, dims));

