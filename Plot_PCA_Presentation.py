"""
The code uses the .csv files of EMCCD_PCA_Shower_PhysProp.py of the GEM and PER showers
The code plot the results for the presentation
"""

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# read the simulated shower .csv file from EMCCD_PCA_Shower_PhysProp.py

df_sim_GEM = pd.read_csv(os.getcwd()+r'\Simulated_GEM.csv')
df_sim_PER = pd.read_csv(os.getcwd()+r'\Simulated_PER.csv')

df_obs_GEM = pd.read_csv(os.getcwd()+r'\GEM.csv')
df_obs_PER = pd.read_csv(os.getcwd()+r'\PER.csv')

# create the dataframe with the simulated shower
df_sim_shower = pd.concat([df_sim_PER,df_sim_GEM], axis=0)

# create the dataframe with the observed shower
df_obs_shower = pd.concat([df_obs_PER,df_obs_GEM], axis=0)

# create the dataframe with all the shower
df_all = pd.concat([df_sim_shower.drop(['rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max'], axis=1),df_obs_shower], axis=0)

# import the selected shower from EMCCD_PCA_Shower_PhysProp.py
df_sel_GEM_PCA = pd.read_csv(os.getcwd()+r'\Simulated_GEM_select_PCA.csv')
df_sel_PER_PCA = pd.read_csv(os.getcwd()+r'\Simulated_PER_select_PCA.csv')

df_sel_shower = pd.concat([df_sel_PER_PCA,df_sel_GEM_PCA], axis=0)

# scale the data
scaler = StandardScaler()
scaler.fit(df_all.drop(['shower_code'], axis=1))

# PCA space of the simulated shower has the same number of PC of the selected shower
pca= PCA(n_components=len(df_sel_shower.columns)-1)
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

# scatter plot of the simulated shower of PC1 vs PC2
sns.scatterplot(x='PC1', y='PC2', data=df_sim_PCA, hue='shower_code', palette=["g","b"], markers=["o", "o"], s=20)
plt.title('PCA space of the simulated GEM & PER shower')
plt.grid(alpha=0.3,linestyle='--')
plt.show()

# scatter plot of both the observed and simulated shower of PC1 vs PC2
sns.scatterplot(x='PC1', y='PC2', data=df_all_PCA, hue='shower_code', palette=["g","b","y","r"], markers=["o", "o","o", "o"], s=20)
plt.title('PCA space of GEM & PER shower')
plt.grid(alpha=0.3,linestyle='--')
plt.show()

# find the mean value of each column with the shower_code GEM and PER
meanPCA = df_all_PCA.groupby('shower_code').mean()

# print only the mean of shower_code GEM and PER
meanPCA = meanPCA.drop(meanPCA.index[2:])
# change the shower_code to GEM_mean and PER_mean
meanPCA = meanPCA.rename(index={'GEM': 'GEM_mean', 'PER': 'PER_mean'})
# add the shower_code to the dataframe
meanPCA['shower_code'] = meanPCA.index

# plot the value of the showers of PC1 and PC2 and the shower mean value
sns.scatterplot(x='PC1', y='PC2', data=df_all_PCA, hue='shower_code', palette=["g","b","y","r"], markers=["o", "o","o", "o"], s=20) 
sns.scatterplot(x='PC1', y='PC2', data=meanPCA, hue='shower_code', palette=["black","black"], marker="D", s=50) 
plt.title('Mean value of the PCA space of the simulated GEM & PER shower')
plt.grid(alpha=0.3,linestyle='--')
plt.show()

# plot the selected meteors in PC1 and PC2 with the shower mean value
sns.scatterplot(x='PC1', y='PC2', data=df_sel_shower, hue='shower_code', palette=["y","r"], markers="X", s=60)
sns.scatterplot(x='PC1', y='PC2', data=df_sim_PCA, hue='shower_code', palette=["g","b"], markers=["o", "o"], s=20)
sns.scatterplot(x='PC1', y='PC2', data=meanPCA, hue='shower_code', palette=["black","black"], marker="D", s=50) 
plt.title('Mean of the PCA of the observed GEM & PER and selected meteors')
plt.grid(alpha=0.3,linestyle='--')
plt.show()

TOT_PCA_DataFrame=pd.concat([df_all_PCA,meanPCA], axis=0)

sns.pairplot(TOT_PCA_DataFrame, hue='shower_code', palette=["g","b","y","r","k","k"], markers=["o", "o","o", "o", "D", "P"], plot_kws={'s': 20},corner=True) #markers=["o", "s", "D", "P"],corner=True 'alpha': 0.6,
plt.show()

TOT_PCA_DataFrame_selected=pd.concat([df_sim_PCA,df_sel_shower,meanPCA], axis=0)

sns.pairplot(TOT_PCA_DataFrame_selected, hue='shower_code', palette=["g","b","y","r","k","k"], markers=["o", "o","P", "D","D", "P"], plot_kws={'s': 20},corner=True) #markers=["o", "s", "D", "P"],corner=True 'alpha': 0.6,
plt.show()
