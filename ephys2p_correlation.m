%% Notes
% import raw imaging data into "unnamed" variable, import raw ephys data
% into "unnnamed1" variable -- change this later

% if using internal reference ch (e.g. AF594 for GCaMP, or AF488 for
% Cal590/Rhod2, import raw reference ch F into "unnamed2" variable

%% variables

prompt = {'ScanImage framerate (Hz)', 'Trigger delay (s)'};
dlgtitle = 'Experiment parameters';
dims = [1 50];
expparameters = str2double(inputdlg(prompt, dlgtitle, dims));  %(1,1) is framerate, (2,1) is delay between image start and ephys

internal = 1;   % 1 for reference ch, 0 if none

%% process ephys data

rawephys = [unnamed1(:,1)/1000 unnamed1(:,2)];
rawdFF = [(unnamed(:,1)/expparameters(1,1)) + expparameters(2,1) unnamed(:,2)]; %creates new data array for dF/F data that applies the sync delay from patch software
if internal == 1
    rawintdFF = [(unnamed2(:,1)/expparameters(1,1)) + expparameters(2,1) unnamed2(:,2)];
end

%% plot data

rawdata = figure(1);
hold on;
yyaxis left
plot(rawephys(:,1), rawephys(:,2));
ylabel('mV');
yyaxis right
plot(rawdFF(:,1), rawdFF(:,2));
plot(rawintdFF(:,1), rawintdFF(:,2));
ylabel('dF/F');
xlabel('Time (s)')
hold off;

% %% set pub figure windows
% prompt = {'Time window start (s)', 'Time window end (s)', 'mV scale min', 'mV scale max', 'dF/F scale min', 'dF/F scale max'};
% dlgtitle = 'Figure properties';
% dims = [1 50];
% figparameters = str2double(inputdlg(prompt, dlgtitle, dims));
% 
% %% plot pub figure
% 
% pubfig = figure(2);
% hold on;
% yyaxis left
% plot(rawephys(:,1), rawephys(:,2));
% ylim([figparameters(3,1) figparameters(4,1)])
% ylabel('mV');
% yyaxis right
% plot(rawdFF(:,1), rawdFF(:,2));
% ylim([figparameters(5,1) figparameters(6,1)])
% ylabel('dF/F');
% xlim([figparameters(1,1) figparameters(2,1)])
% xlabel('Time (s)')
% hold off;
