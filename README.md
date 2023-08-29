# Meteor_PCA
Using Principal Components Analysis (PCA) on a dataset of faint meteors from EMCCDs, researchers map physical properties between observed and synthetic meteor data. The method estimates the average properties of meteor showers, with results aligning with prior studies. Preliminary results reveal distinct physical characteristics of Geminids and Perseids meteor showers. PCA's effectiveness suggests it can narrow uncertainties, aiding future machine learning applications for specific meteor event analysis.
- EMCCD_PCA_Shower_PhysProp.py = creates from raw .json files the .csv data and select the best simulated meteors for the selected shower/s
- Plot_PCA_Presentation.py = is the code used to develope the plots for the final presentation
- Plot_selected_range.m = is the code used for the range distribution plots
- .csv files are created by EMCCD_PCA_Shower_PhysProp.py
- .zip file is GenerateSimulations.py modified to genereate EMCCD data
