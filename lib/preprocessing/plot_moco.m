function plot_moco(file_rp, path_output, name_output)
% This function plots the motion parameter from the SPM12 realignment
% processing. Separate plots for translation (in mm) and rotation (in deg)
% are saved.
% Inputs:
    % file_rp: file name of textfile with realignment parameters.
    % path_output: path where output is written.
    % name_output: basename of saved plots.
    
% created by Daniel Haenelt
% Date created: 23-02-2020
% Last modified: 23-02-2020

% make output folder
if ~exist(path_output,'dir') 
    mkdir(path_output); 
end

% read realignment parameters
M = dlmread(file_rp);

% rad2deg
M(:,4) = radtodeg(M(:,4));
M(:,5) = radtodeg(M(:,5));
M(:,6) = radtodeg(M(:,6));

% translational displacement
M_trans = sqrt(M(:,1).^2+M(:,2).^2+M(:,3).^2);

% rotational displacement
M_rot = sqrt(M(:,4).^2+M(:,5).^2+M(:,6).^2);

transFig = figure('visible','off');
hold on
plot(M(:,1));
plot(M(:,2));
plot(M(:,3));
plot(M_trans,'LineWidth',2);
title(['Translational movement in session ' name_output],'Interpreter','None');
xlabel('number of volume');
ylabel('Translation in mm');
legend('x','y','z','total translation');
saveas(gcf,fullfile(path_output,[name_output '_trans.png']));
close(transFig);

radFig = figure('visible','off');
hold on
plot(M(:,4));
plot(M(:,5));
plot(M(:,6));
plot(M_rot,'LineWidth',2);
title(['Rotational movement in session ' name_output],'Interpreter','None');
xlabel('number of volume');
ylabel('Rotation in deg');
legend('pitch','roll','yaw','total rotation');
saveas(gcf,fullfile(path_output,[name_output '_rot.png']));
close(radFig);