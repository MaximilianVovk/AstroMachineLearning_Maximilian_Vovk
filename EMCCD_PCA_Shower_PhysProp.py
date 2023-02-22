"""
The code is used to extract the physical properties of the simulated showers from EMCCD observations
by selecting the most similar simulated shower.
The code is divided in three parts:
    1. from GenerateSimulations.py output folder extract the simulated showers observable and physiscal characteristics
    2. extract from the EMCCD solution_table.json file the observable property of the shower
    3. select the simulated meteors similar to the EMCCD meteor observations and extract their physical properties
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import seaborn as sns
import scipy.spatial.distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# MODIFY HERE THE PARAMETERS ###############################################################################


# Set the shower name (can be multiple) e.g. 'GEM' or ['PER', 'GEM']
Shower=['PER', 'GEM']

# Set the folder where are the GenerateSimulations.py output json files e.g. "Simulations_"+Shower+""
# the numbr of showers and folder must be the same
folder_GenerateSimulations_json = ["Simulations_"+f+"" for f in Shower]

# save the extracted dataframe in a .csv file so to avoid to re-run the code
save_it=True

# Set the number of selected metoers to extract the physical properties
n_selected=50

# FUNCTIONS ###########################################################################################


def read_GenerateSimulations_folder_output(shower_folder,Shower='',save_it=False):
    ''' 
    It reads the GenerateSimulations.py output json files from the shower_folder and extract the observable and physical property
    The values are given in a dataframe format and if requestd are saved in a .csv file called Shower+".csv"
    Keyword arguments:
    shower_folder:  folder of the simulated meteors.
    Shower:         Shower name, by default there is no name.
    save_it:        Boolean - save the extracted dataframe in a .csv, by default it is not saved.
    '''

    # open the folder and extract all the json files
    os.chdir(shower_folder)
    extension = 'json'
    all_jsonfiles = [i for i in glob.glob('*.{}'.format(extension))]

    # Initialized the values of the dataframe
    dataList = [['', 0, 0, 0,\
        0, 0, 0, 0, 0, 0,\
        0, 0, 0, 0, 0, 0,\
        0, 0, 0,\
        0, 0]]

    # create a dataframe to store the data
    df_json = pd.DataFrame(dataList, columns=['shower_code','vel_init_norot','vel_avg_norot','duration',\
        'mass','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag',\
        'F','trail_len','acceleration','zenith_angle', 'rho','sigma',\
        'erosion_height_start','erosion_coeff', 'erosion_mass_index',\
        'erosion_mass_min','erosion_mass_max'])

    # open the all file and extract the data
    for i in range(len(all_jsonfiles)):
        f = open(all_jsonfiles[i],"r")
        data = json.loads(f.read())

        # show the current processed file and the number of files left to process
        print(all_jsonfiles[i]+' - '+str(len(all_jsonfiles)-i)+' left')

        if data['ht_sampled']!= None: 
            # from 'params' extract the observable parameters and save them in a list
            mass = data['params']['m_init']['val']
            vel_init_norot = data['params']['v_init']['val']/1000
            zenith_angle= data['params']['zenith_angle']['val']*180/np.pi

            # the observable parameters below are calculated from the sampled data
            duration = data['time_sampled'][-1]
            begin_height = data['ht_sampled'][0] / 1000
            end_height = data['ht_sampled'][-1] / 1000
            peak_abs_mag = data['mag_sampled'][np.argmin(data['mag_sampled'])]
            F = (begin_height - (data['ht_sampled'][np.argmin(data['mag_sampled'])] / 1000)) / (begin_height - end_height)
            beg_abs_mag	= data['mag_sampled'][0]
            end_abs_mag	= data['mag_sampled'][-1]
            trail_len = data['len_sampled'][-1] / 1000
            shower_code = 'sim_'+Shower
            vel_avg_norot = trail_len / duration
            acceleration = (vel_init_norot - vel_avg_norot)/duration

            # from 'params' extract the physical parameters and save them in a list
            rho = data['params']['rho']['val']
            sigma = data['params']['sigma']['val']
            erosion_height_start = data['params']['erosion_height_start']['val']/1000
            erosion_coeff = data['params']['erosion_coeff']['val']
            erosion_mass_index = data['params']['erosion_mass_index']['val']
            erosion_mass_min = data['params']['erosion_mass_min']['val']
            erosion_mass_max = data['params']['erosion_mass_max']['val']

            # add a new line in dataframe
            df_json.loc[len(df_json)] = [shower_code, vel_init_norot, vel_avg_norot, duration,\
            mass, begin_height, end_height, peak_abs_mag, beg_abs_mag, end_abs_mag,\
            F, trail_len, acceleration, zenith_angle, rho, sigma,\
            erosion_height_start, erosion_coeff, erosion_mass_index,\
            erosion_mass_min, erosion_mass_max]

    print('succesfully read all the files \n \n')

    # delete the first line of the dataframe that is empty
    df_json = df_json.drop([0])
    
    # go back to the initial directrory
    os.chdir('..')
    # save the dataframe in a csv file in the same folder of the code withouth the index
    if save_it == True:
        df_json.to_csv(os.getcwd()+r'\Simulated_'+Shower+'.csv', index=False)

    f.close()

    return df_json





def read_solution_table_json(df_simul,Shower='',save_it=False):
    '''
    It reads the solution_table.json file and extract the observable property from the EMCCD camera results
    The values are given in a dataframe format and if requestd are saved in .csv file called "Simulated_"+Shower+".csv"
    Keyword arguments:
    df_simul:       dataframe of the simulated shower
    Shower:         Shower name, by default there is no name.
    save_it:        Boolean - save the extracted dataframe in a .csv, by default it is not saved.
    '''

    # open the solution_table.json file
    f = open('solution_table.json',"r")
    data = json.loads(f.read())
    # create a dataframe to store the data
    df = pd.DataFrame(data, columns=['shower_code','vel_init_norot','vel_avg_norot','vel_init_norot_err','beg_fov','end_fov','elevation_norot','duration','mass','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag','F'])
    
    # select the data from the shower EMCCD database and delete possible wrong values or data that are hard to simulate for GenerateSimulations code
    df_shower_EMCCD = df.loc[
    (df.shower_code == Shower) & (df.beg_fov) & (df.end_fov) & (df.vel_init_norot_err < 2) & (df.begin_height > df.end_height) & (df.elevation_norot>15) &
    (df.elevation_norot >=0) & (df.elevation_norot <= 95) & (df.begin_height < 180) & (df.F > 0) & (df.F < 1) & (df.begin_height > 80) * (df.vel_init_norot < 75)
    ]
    # print the total number of observations
    print(len(df_shower_EMCCD),' number of ',Shower,' meteor in solution_table')

    # delete all the outliers in the shower data and consider only the one within the simulated data range
    df_shower_EMCCD_no_outliers = df_shower_EMCCD.loc[
    (df_shower_EMCCD.mass<np.max(df_simul['mass'])) & (df_shower_EMCCD.mass>np.min(df_simul['mass'])) &
    (df_shower_EMCCD.mass<np.percentile(df_shower_EMCCD['mass'], 90)) & (df_shower_EMCCD.mass>np.percentile(df_shower_EMCCD['mass'], 10)) &
    (df_shower_EMCCD.duration<np.max(df_simul['duration'])) & (df_shower_EMCCD.duration>np.min(df_simul['duration'])) &
    (df_shower_EMCCD.duration<np.percentile(df_shower_EMCCD['duration'], 90)) & (df_shower_EMCCD.duration>np.percentile(df_shower_EMCCD['duration'], 10)) &
    (df_shower_EMCCD.beg_abs_mag<np.max(df_simul['beg_abs_mag'])) & (df_shower_EMCCD.beg_abs_mag>np.min(df_simul['beg_abs_mag'])) &
    (df_shower_EMCCD.beg_abs_mag<np.percentile(df_shower_EMCCD['beg_abs_mag'], 90)) & (df_shower_EMCCD.beg_abs_mag>np.percentile(df_shower_EMCCD['beg_abs_mag'], 10)) &
    (df_shower_EMCCD.vel_init_norot<np.max(df_simul['vel_init_norot'])) & (df_shower_EMCCD.vel_init_norot>np.min(df_simul['vel_init_norot'])) &
    (df_shower_EMCCD.vel_init_norot<np.percentile(df_shower_EMCCD['vel_init_norot'], 90)) & (df_shower_EMCCD.vel_init_norot>np.percentile(df_shower_EMCCD['vel_init_norot'], 10))
    ]
    # print the number of droped observation
    print(len(df_shower_EMCCD)-len(df_shower_EMCCD_no_outliers),'number of droped ',Shower,' observation \n \n')

    # trail_len in km
    df_shower_EMCCD_no_outliers.insert(len(df_shower_EMCCD_no_outliers.columns), "trail_len", (df_shower_EMCCD_no_outliers['begin_height'] - df_shower_EMCCD_no_outliers['end_height'])/np.sin(np.radians(df_shower_EMCCD_no_outliers['elevation_norot'])), True)
    # acceleration in km/s^2
    df_shower_EMCCD_no_outliers.insert(len(df_shower_EMCCD_no_outliers.columns), "acceleration", (df_shower_EMCCD_no_outliers['vel_init_norot'] - df_shower_EMCCD_no_outliers['vel_avg_norot'])/(df_shower_EMCCD_no_outliers['duration']), True)
    # Zenith angle in radians
    df_shower_EMCCD_no_outliers.insert(len(df_shower_EMCCD_no_outliers.columns), "zenith_angle", (90 - df_shower_EMCCD_no_outliers['elevation_norot']), True)

    # delete the columns that are not needed
    df_shower_EMCCD_no_outliers = df_shower_EMCCD_no_outliers.drop(['vel_init_norot_err','beg_fov','end_fov','elevation_norot'], axis=1)

    # save the dataframe in a csv file in the same folder of the code withouth the index
    if save_it == True:
        df_shower_EMCCD_no_outliers.to_csv(os.getcwd()+r'\\'+Shower+'.csv', index=False)

    f.close()

    return df_shower_EMCCD_no_outliers






# CODE ####################################################################################

# save all the simulated showers in a list
df_sim_shower = []
df_obs_shower = []
# search for the simulated showers in the folder
for current_shower in Shower:
    # check in the current folder there is a csv file with the name of the simulated shower
    if os.path.isfile(os.getcwd()+r'\Simulated_'+current_shower+'.csv'):
        # if yes read the csv file
        df_sim = pd.read_csv(os.getcwd()+r'\Simulated_'+current_shower+'.csv')
    else:
        # if no read the json files in the folder and create a new csv file
        df_sim = read_GenerateSimulations_folder_output(folder_GenerateSimulations_json[Shower.index(current_shower)],current_shower,save_it)

    # append the simulated shower to the list
    df_sim_shower.append(df_sim)

    if os.path.isfile(os.getcwd()+r'\\'+current_shower+'.csv'):
        # if yes read the csv file
        df_obs = pd.read_csv(os.getcwd()+r'\\'+current_shower+'.csv')
    else:
        # if no read the solution_table.json file
        df_obs = read_solution_table_json(df_sim,current_shower,save_it)

    # append the observed shower to the list
    df_obs_shower.append(df_obs)
    

# concatenate all the simulated shower in a single dataframe
df_sim_shower = pd.concat(df_sim_shower)

# concatenate all the EMCCD observed showers in a single dataframe
df_obs_shower = pd.concat(df_obs_shower)

# concatenate all the observation data in a single dataframe
df_all = pd.concat([df_sim_shower.drop(['rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max'], axis=1),df_obs_shower], axis=0)

# Now we have all the data and we apply PCA to the dataframe

sns.pairplot(df_all[['shower_code','mass','duration','beg_abs_mag','vel_init_norot']], hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
plt.show()

# scale the data
scaler = StandardScaler()
scaler.fit(df_all.drop(['shower_code'], axis=1))

# find hyper plane that separates the data
pca= PCA()
pca.fit(scaler.transform(df_all.drop(['shower_code'], axis=1)))

# transform the data to the new PCA space
all_PCA = pca.transform(scaler.transform(df_all.drop(['shower_code'], axis=1)))

# PLOT explained variance ratio #########################################

# compute the explained variance ratio
percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
print("explained variance ratio: \n",percent_variance)

# name of the principal components
columns_PC = ['PC' + str(x) for x in range(1, len(percent_variance)+1)]

# plot the explained variance ratio of each principal componenets base on the number of column of the original dimension
plt.bar(x= range(1,len(percent_variance)+1), height=percent_variance, tick_label=columns_PC, color='black')
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.show()

# PLOT the correlation coefficients #########################################

# Compute the correlation coefficients
cov_data = pca.components_.T

# Plot the correlation matrix
img = plt.matshow(cov_data.T, cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
plt.colorbar(img)
# name of the variable on the x-axis in a short form
rows=['init.vel','avg.vel','durat','mass','beg.height','end.height','mag.peak','beg mag.','end mag.','F','trail len.','acc.','zenith ang.']

# Add the variable names as labels on the x-axis and y-axis
plt.xticks(range(len(rows)), rows, rotation=90)
plt.yticks(range(len(columns_PC)), columns_PC)

# plot the influence of each component on the original dimension
for i in range(cov_data.shape[0]):
    for j in range(cov_data.shape[1]):
        plt.text(i, j, "{:.2f}".format(cov_data[i, j]), size=12, color='black', ha="center", va="center")   
plt.show()

# PLOT the shorter PCA space ########################################

# find the number of PC that explain 95% of the variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# recomute PCA with the number of PC that explain 95% of the variance
pca= PCA(n_components=np.argmax(cumulative_variance >= 0.95) + 1)
pca.fit(scaler.transform(df_all.drop(['shower_code'], axis=1)))
# recompute the PCA space for the new number of PC
all_PCA = pca.transform(scaler.transform(df_all.drop(['shower_code'], axis=1)))

# select only the column with in columns_PC with the same number of n_components
columns_PC = ['PC' + str(x) for x in range(1, pca.n_components_+1)]

# create a dataframe with the PCA space
df_all_PCA = pd.DataFrame(data = all_PCA, columns = columns_PC)
# add the shower code to the dataframe
df_all_PCA['shower_code'] = df_all['shower_code'].values

# delete the lines after len(df_sim_shower) to have only the simulated shower
df_sim_PCA = df_all_PCA.drop(df_all_PCA.index[len(df_sim_shower):])



# plot all the data in the PCA space
sns.pairplot(df_all_PCA, hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
plt.show()

##########################################
# define the mean position and extract the n_selected meteor closest to the mean

# find the mean base on the shower_code in the PCA space
meanPCA = df_all_PCA.groupby('shower_code').mean()

# create a list with the selected meteor properties and PCA space
df_sel_shower=[]
df_sel_PCA=[]
for current_shower in Shower:
    # find the mean of the simulated shower
    meanPCA_current = meanPCA.loc[(meanPCA.index == current_shower)]
    # meanPCA_current = meanPCA_current.values[0]

    # find the distance between the mean and all the simulated shower
    distance_current = []
    for i in range(len(df_sim_shower)):
        distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current, all_PCA[i]))
    
    # create an index with lenght equal to the number of simulations and set all to false
    index_current = [False]*(len(df_sim_shower))
    for i in range(n_selected):
        # define the index of the n_selected closest to the mean
        index_current[distance_current.index(min(distance_current))]=True
        distance_current[distance_current.index(min(distance_current))]=1000

    # create a dataframe with the selected simulated shower in the PCA space
    df_PCA_selected = df_sim_PCA[index_current]
    # delete the shower code
    df_PCA_selected = df_PCA_selected.drop(['shower_code'], axis=1)
    # add the shower code
    df_PCA_selected.insert(0, "shower_code", current_shower+'_sel', True)
    # append the simulated shower to the list
    df_sel_PCA.append(df_PCA_selected)

    # save the dataframe to a csv file withouth the index
    df_PCA_selected.to_csv(os.getcwd()+r'\Simulated_'+current_shower+'_select_PCA.csv', index=False)

    # create a dataframe with the selected simulated shower characteristics
    df_sim_selected = df_sim_shower[index_current]
    # delete the shower code
    df_sim_selected = df_sim_selected.drop(['shower_code'], axis=1)
    # add the shower code
    df_sim_selected.insert(0, "shower_code", current_shower+'_sel', True)
    # append the simulated shower to the list
    df_sel_shower.append(df_sim_selected)

    # save the dataframe to a csv file withouth the index
    df_sim_selected.to_csv(os.getcwd()+r'\Simulated_'+current_shower+'_select.csv', index=False)

# concatenate the list of the PC components to a dataframe
df_sel_PCA = pd.concat(df_sel_PCA)

# concatenate the list of the properties to a dataframe
df_sel_shower = pd.concat(df_sel_shower)

# PLOT the selected simulated shower ########################################

# dataframe with the simulated and the selected meteors in the PCA space
df_sim_sel_PCA = pd.concat([df_sim_PCA,df_sel_PCA], axis=0)

sns.pairplot(df_sim_sel_PCA, hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
plt.show()


# dataframe with the simulated and the selected meteors physical characteristics
df_sim_sel_shower = pd.concat([df_sim_shower,df_sel_shower], axis=0)

sns.pairplot(df_sim_sel_shower[['shower_code','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']], hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
plt.show()
    