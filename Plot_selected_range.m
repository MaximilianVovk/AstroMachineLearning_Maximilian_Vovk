clc
clear all
close all
%% Showers

% shower name
Shower='GEM'

Shower='PER'

%% DATA IMPORT

Shower_sel = readtable(append('Simulated_',Shower,'_select.csv'));
Shower_sim = readtable(append('Simulated_',Shower,'.csv'));

% take the data of the physical parameters
variables_coef_sel = Shower_sel{:,15:21};
variables_coef_sim = Shower_sim{:,15:21};

% columns names in an array
colum_name=Shower_sel.Properties.VariableNames;

%% PLOT the range of values cover for the presentation

figure(1)
for ii=1:min(size(variables_coef_sel))   

    if Shower=='GEM'
        colorShower='r';
    elseif Shower=='PER'
        colorShower='y';
    end
    % create a subplot entry for each component
    subplot(7,1,ii)
    area([prctile(variables_coef_sel(:,ii),10) prctile(variables_coef_sel(:,ii),90)],[1 1],'FaceColor',colorShower,'EdgeColor',colorShower);
    title(colum_name{14+ii},'FontSize',10, 'Interpreter', 'none','fontweight','normal')
    
    % delete any y labels
    set(gca,'YTickLabel',[]);
    % set the x limit equal to the simulated data
    xlim([min(variables_coef_sim(:,ii)) max(variables_coef_sim(:,ii))])

    % use a logaritmic scale for part of the data
    if ii>=4 || ii==2
        set(gca, 'Xscale', 'log');
    end
    if Shower=='GEM'
        sgtitle('Geminids','fontweight','bold')
    elseif Shower=='PER'
        sgtitle('Perseids','fontweight','bold')
    end
end