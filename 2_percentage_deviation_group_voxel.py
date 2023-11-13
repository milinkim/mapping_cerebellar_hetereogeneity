#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:55:40 2022

@author: Milin Kim

"""
import random 
import glob
import pandas
import os
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joypy
from sklearn.model_selection import train_test_split
from pcntoolkit.normative import estimate, evaluate
from pcntoolkit.util.utils import create_bspline_basis, compute_MSLL
import pcntoolkit as pcn
import pickle
import SUITPy.flatmap as flatmap
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from colorspacious import cspace_converter

###########################load directories and data###########################
#change directory 
cd /s_normative_model/s_suit_nm_5/s_estimate_clinical/s_estimate_Z/
figure_dir="s_normative_model/figures/"

#read Z estimates for all voxels 
frames = []
for file in os.listdir(os.getcwd()):
    if file.endswith('.pkl'):
        files = pd.read_pickle(file)
        frames.append(files)
master_df = pd.concat(frames, axis = 1)
#column names into integeder
master_df.columns=master_df.columns.map(int)
master_df=master_df.reindex(sorted(master_df.columns), axis=1)
master_df.columns=master_df.columns.map(str)

#concat diagnosis
suit_hc_clinical_te=pd.read_pickle('/s_normative_model/s_suit_split_chunk/suit_chunk_1_te_clinical.pkl')
master_df=master_df.reset_index()
df_te=suit_hc_clinical_te[['ID', 'diagnosis', 'site']]
df_te=df_te.reset_index()
df_Z_te=pd.merge(master_df, df_te, left_index=True, right_index=True)


#groupby diagnosis/simplify
def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]
#count diagnosis
df_Z_te.groupby('diagnosis').size()
df_Z_te['diagnosis'].unique()


#group them into diagnosis df
num_clinical = 2 #multiple comparison 
hc=df_Z_te[df_Z_te['diagnosis']=='HC'] #hc
disorder1=df_Z_te[df_Z_te['diagnosis']=='disorder1'] 
disorder2=df_Z_te[df_Z_te['diagnosis']=='disorder2']

df=[]
groups=[hc, disorder1, disorder2]
groups_s=['hc', 'disorder1', 'disorder2']

    
clinical=pd.concat(groups)

#groups=[hc]
#groups_s=['hc']

wdir="/s_normative_model/s_suit_nm_5/percent_deviation/"
#load mask and example image and test
mask_image='/suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr_bin_mul.nii' #mask thresholded at 0.2
example_image='suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr.nii'


#######################plot extreme deviation #######################
for i,j in zip(groups, groups_s):
    c=i
    c_name=j
    
    file_name_p=c_name+'_Z_p'
    file_name_n=c_name+'_Z_n'
    c=c.set_index('ID')
    del c['diagnosis']
    del c['site']
    
    df=[]
    for i in c.columns: #loop per voxel
        i_df= np.array(c[i]) 
        p_outliers=0
        n_outliers=0
        n_subject=len(c.index)
        for j in i_df: #go through each Z scores if above or below +/-1.96 
            if j >= 1.96:
                p_outliers+=1
            elif j <= -1.96:
                n_outliers+=1
      
        df.append((i, p_outliers, p_outliers/n_subject, n_outliers, n_outliers/n_subject))  
    c_Z=pd.DataFrame(df, columns=('voxel', 'num_p_outliers', 'p_p_outliers','num_n_outliers', 'p_n_outliers'))
    c_Z_p=c_Z['p_p_outliers'] 
    c_Z_p=c_Z_p.to_numpy()
    c_Z_n=c_Z['p_n_outliers'] 
    c_Z_n=c_Z_n.to_numpy()
   
    #put it in nifti for positive/negative
    p_nii=pcn.dataio.fileio.save_nifti(c_Z_p, wdir+file_name_p+'.nii', examplenii=example_image, mask=mask_image)
    n_nii=pcn.dataio.fileio.save_nifti(c_Z_n, wdir+file_name_n+'.nii', examplenii=example_image, mask=mask_image)
    plt.gcf().set_dpi(300)
    funcdata_p = flatmap.vol_to_surf(wdir+file_name_p+'.nii',space='SUIT')
    flatmap.plot(data=funcdata_p, cmap='hot', threshold=[0.00, 0.2], cscale=[0,.10],bordersize=1.5, bordercolor='w', new_figure=True,colorbar=True, render='matplotlib')
    plt.savefig(os.path.join(figure_dir,file_name_p +'_10.png'))
    plt.gcf().set_dpi(300)
    funcdata_n = flatmap.vol_to_surf(wdir+file_name_n+'.nii',space='SUIT')
    flatmap.plot(data=funcdata_n, cmap='hot', threshold=[0.00, 0.2], cscale=[0,.10], bordersize=1.5, bordercolor='w', new_figure=True,colorbar=True, render='matplotlib')
    plt.savefig(os.path.join(figure_dir,file_name_n +'_10.png'))



##########################calculate nonparametric comparison ##################

for i,j in zip(groups, groups_s):
    c=i
    c_name=j
    del c['diagnosis']
    del c['site']
    c.set_index('ID', inplace=True)
    del hc['diagnosis']
    del hc['site']
    hc.set_index('ID', inplace=True)
    
    presult=[]
    for i in c.columns: #loop participant
            i_df= np.array(c[i])
            i_hc_df= np.array(hc[i])
            series_c = pd.Series(i_df) 
            series_hc = pd.Series(i_hc_df) 
            #median
            median_c=round(series_c.median(),3)
            median_hc=round(series_hc.median(),3)
            #calculate mann-whiteney
            effect=pg.mwu(series_hc,series_c, alternative='two-sided')
            presult.append((i,c_name,effect['U-val'][0],effect['p-val'][0],effect.RBC[0],effect.CLES[0]))
        
    df_df2=pd.DataFrame(presult, columns=('Voxels','Diagnosis','U_value', 'P-value','RBC','CLES'))
    #df_df2.to_csv('/s_normative_model/s_suit_nm_5/group_comp/non_parametric_voxels_'+c_name+'.csv')
    
    #plot on cerebellum
    file_name= c_name+'non_para_voxels' 
    row_index = 142978  # Change this to the row index you want to modify
    
    for r in range(row_index): #multiple correction 142978*num_clinical*2
        if df_df2.at[r, 'P-value'] >= (.05):
            df_df2.at[r, 'RBC'] = 0
    
    c_Z = df_df2['RBC'].astype(float).values
    
    wdir = "s_normative_model/s_suit_nm_5/group_comp/"
    nii = pcn.dataio.fileio.save_nifti(c_Z, wdir+file_name+'_non_para.nii', examplenii=example_image, mask=mask_image)
    plt.gcf().set_dpi(300)
    funcdata_e = flatmap.vol_to_surf(wdir+file_name+'_non_para.nii',space='SUIT')
    flatmap.plot(data=funcdata_e, cmap='hot', threshold=[0,1], cscale=[0,.3],bordersize=1, bordercolor='w', new_figure=True,colorbar=True, render='matplotlib')
    plt.savefig(os.path.join(figure_dir,file_name +'_non_para_pos_uncorrected.png')) 


  