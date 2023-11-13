#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:11:03 2023

@author: Milin Kim
"""
import glob
import os
import pandas as pd
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
from scipy.stats import mannwhitneyu
from numpy import mean,std
from statistics import mean, stdev
from math import sqrt
import pingouin as pg

''' #flirt atlases 
FSL: flirt -in acapulco/output/t1/acapulco_labels_name_without_wm.nii -ref suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr_bin_mul.nii -omat affine.mat
FSL: flirt -in acapulco/output/t1/acapulco_labels_name.nii -datatype int -ref suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr_bin_mul.nii -interp nearestneighbour -init affine.mat -applyxfm  -out suit/mask_cerebellum_binary/acapulco_in_suit_mat.nii.gz
mm=pcn.dataio.fileio.load_nifti(mask_image)
mm = pd.DataFrame(mm)

FSL: flirt -in acapulco/output/t1/MDTB_10Regions.nii -ref suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr_bin_mul.nii -omat affine.mat
FSL: flirt -in acapulco/output/t1/MDTB_10Regions.nii -datatype int -ref suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr_bin_mul.nii -interp nearestneighbour -init affine.mat -applyxfm  -out suit/mask_cerebellum_binary/acapulco_in_mdtb.nii.gz

FSL: flirt -in acapulco/output/t1/Buckner_17Networks.nii -ref suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr_bin_mul.nii -omat affine.mat
FSL: flirt -in acapulco/output/t1/Buckner_17Networks.nii -datatype int -ref suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr_bin_mul.nii -interp nearestneighbour -init affine.mat -applyxfm  -out suit/mask_cerebellum_binary/acapulco_in_buckner.nii.gz

FSL: flirt -in acapulco/output/t1/Lobules-SUIT-prob.nii -ref suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr_bin_mul.nii -omat affine.mat
FSL: flirt -in acapulco/output/t1/Lobules-SUIT-prob.nii -datatype int -ref suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr_bin_mul.nii -interp nearestneighbour -init affine.mat -applyxfm  -out suit/mask_cerebellum_binary/acapulco_in_suit.nii.gz
''' 
###########################load atlases and data####################################
acapulco_p='/suit/mask_cerebellum_binary/acapulco_in_suit_mat.nii.gz' #mask thresholded at 0.2
mdtb_p='/suit/mask_cerebellum_binary/acapulco_in_mdtb.nii.gz'
buckner_p='suit/mask_cerebellum_binary/acapulco_in_buckner.nii.gz'
wdir="s_normative_model/s_suit_nm_5/"
mask_image='suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr_bin_mul.nii' #mask thresholded at 0.2
example_image='suit/mask_cerebellum_binary/Merge_suit_1_mean_nan_thr.nii'



#########load atlas
flirt_image=mdtb_p #change#######
atlas_name='mdtb'####change
roi=10*2*5###change (roi x direciton x diagnosis) for multiple correction
mul=pcn.dataio.fileio.load_nifti(flirt_image, mask=mask_image)
mul = pd.DataFrame(mul)
mul2=mul[0].replace(to_replace=0, method='ffill').values
mul2=mul[0].unique()
binarize_list=np.delete(mul2, np.where(mul2 == 0))
print(binarize_list) #check roi 

#####load z score 
directory='/s_normative_model/s_suit_nm_5/s_estimate_clinical/s_estimate_Z/'
os.chdir(directory)
figure_dir="/s_normative_model/s_figures/"

#####read Z estimates for all voxels 
frames = []
for file in os.listdir(os.getcwd()):
    if file.endswith('.pkl'):
        files = pd.read_pickle(file)
        frames.append(files)
master_df = pd.concat(frames, axis = 1)
#column names into integeder
master_df.columns=master_df.columns.map(int)
#sort columns 
master_df=master_df.reindex(sorted(master_df.columns), axis=1)
master_df.columns=master_df.columns.map(str)

#concat diagnosis
suit_hc_clinical_te=pd.read_pickle('s_normative_model/s_suit_split_chunk/suit_chunk_1_te_clinical.pkl')
master_df=master_df.reset_index()
df_te=suit_hc_clinical_te[['ID', 'diagnosis', 'site']]
df_te=df_te.reset_index()
df_Z_te=pd.merge(master_df, df_te, left_index=True, right_index=True)
df_Z_te=df_Z_te.rename(columns={'ID_x':'ID' })
del df_Z_te['ID_y']
del df_Z_te['index']

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

     
c_groups=[disorder1, disorder2]
c_groups_s=['disorder1', 'disorder2']


for c in c_groups:
    c.set_index('ID', inplace=True)
    del c['diagnosis']
    del c['site']
    #del c['sex']#
    #del c['age']#
    
del hc['diagnosis']
del hc['site']
hc.set_index('ID', inplace=True)
#asd.set_index('ID', inplace=True)
######loop from here

presult=[]

for c, c_name in zip (c_groups, c_groups_s):
    hc_name='hc_all'
    c_hc=hc
    c_hc_t=c_hc.T #(142981, #of clinical/hc)
    c_t=c.T
    
    for l in binarize_list: #loop through each lobules
        #binarize_t=binarize.T
        mul3=mul[0].replace(to_replace=0, method='ffill').values
        mul3=mul3.astype(float)
        binarize=mul3
        binarize[binarize!=l]=0
        binarize[binarize==l]=1
        binarize=binarize.T
        #binarize areas that are roi
        c_hc_t_b=c_hc_t * binarize[:,None]
        c_t_b=c_t * binarize[:,None]
        
        #get voxel number for each roi
        n_lobules=np.unique(binarize, return_counts=True)
        n_lobules = [tup[1] for tup in n_lobules]
        v_lobules=n_lobules[1] #lobule number of voxel number 
        
        #get rid of rows with zero
        c_t_z=c_t_b[~(c_t_b == 0).all(axis=1)]
        c_hc_t_z=c_hc_t_b[~(c_hc_t_b == 0).all(axis=1)]
        
        #####get positive/negative deviations for clinical 
        df=[]
        for i in c_t_z.columns: #loop participant
            i_df= np.array(c_t_z[i]) 
            p_outliers=0
            n_outliers=0
            n_subject=len(c_t_z.index)
            for j in i_df: #go through each Z scores 
                if j >= 1.96:
                    p_outliers+=1
                elif j <= -1.96:
                    n_outliers+=1
            #print(f'number of outliers for {i}:', n_outliers, 'percentage:',n_outliers/142978 )
            df.append((i, p_outliers, p_outliers/v_lobules, n_outliers, n_outliers/v_lobules))
        
            #print(f'number of outliers for {i}:', n_outliers, 'percentage:',n_outliers/142978 )
        c_Z=pd.DataFrame(df, columns=('participant', 'num_p_outliers', 'p_p_outliers','num_n_outliers', 'p_n_outliers'))
         
        c_Z_p=c_Z['p_p_outliers'] 
        c_Z_p=c_Z_p.to_numpy()
        c_Z_n=c_Z['p_n_outliers'] 
        c_Z_n=c_Z_n.to_numpy()
        
        #repeat the same thing for hc 
        hc_df=[]
        for i in c_hc_t_z.columns: #loop participant
            i_df= np.array(c_hc_t_z[i]) 
            p_outliers=0
            n_outliers=0
            n_subject=len(c_hc_t_z.index)
            for j in i_df: #go through each Z scores 
                if j >= 1.96:
                    p_outliers+=1
                elif j <= -1.96:
                    n_outliers+=1
            #print(f'number of outliers for {i}:', n_outliers, 'percentage:',n_outliers/142978 )
            hc_df.append((i, p_outliers, p_outliers/v_lobules, n_outliers, n_outliers/v_lobules))
        
            #print(f'number of outliers for {i}:', n_outliers, 'percentage:',n_outliers/142978 )
        hc_Z=pd.DataFrame(hc_df, columns=('participant', 'num_p_outliers', 'p_p_outliers','num_n_outliers', 'p_n_outliers'))
        
        #put percentage of outliers to series
        hc_Z_p=hc_Z['p_p_outliers'] 
        hc_Z_p=hc_Z_p.to_numpy()
        hc_Z_n=hc_Z['p_n_outliers'] 
        hc_Z_n=hc_Z_n.to_numpy()
        series_p = pd.Series(c_Z_p) 
        series_p_hc = pd.Series(hc_Z_p) 
        series_n = pd.Series(c_Z_n)
        series_n_hc = pd.Series(hc_Z_n) 
        
         #get mean and std
        median_p=round(series_p.median(),3)
        #std_p=round(series_p.std(),3)
        median_hc_p=round(series_p_hc.median(),3)
        #std_hc_p=round(series_p_hc.std(),3)
        median_n=round(series_n.median(),3)
        #std_n=round(series_n.std(),3)
        median_hc_n=round(series_n_hc.median(),3)
        #std_hc_n=round(series_n_hc.std(),3)
        
        n_effect=pg.mwu(series_n_hc,series_n, alternative='two-sided')
        presult.append((l,c_name,n_effect['U-val'][0],n_effect['p-val'][0],'negative',n_effect.RBC[0],n_effect.CLES[0],median_n, median_hc_n, c_t_z.shape[1], c_hc_t_z.shape[1]))
        
        p_effect=pg.mwu(series_p_hc,series_p, alternative='two-sided')
        presult.append((l,c_name,p_effect['U-val'][0],p_effect['p-val'][0],'positive',p_effect.RBC[0],p_effect.CLES[0],median_p, median_hc_p,c_t_z.shape[1], c_hc_t_z.shape[1]))
        
df_df2=pd.DataFrame(presult, columns=('ROI','Diagnosis','U_value', 'P-value','p_or_n','RBC','CLES','Median Clinical','Median Control', 'N Clinical', 'N Control'))
df_df2.to_csv('/s_figures/deviation_per_parti_'+atlas_name+'_'+'_allhc2.csv')

##########plot effect size########################################
seriesmap=mul2
grp=df_df2.groupby(['Diagnosis'])
def str2bool(v):
  return v.lower() in ('true')
df_r = []


for m in c_groups_s:
    selected_group = grp.get_group(m)  # diagnosis group
    #get atlas
    seriesmap_p = seriesmap
    seriesmap_n = seriesmap
    
    #group by either positive or negative deviation
    grp_pn = selected_group.groupby(['p_or_n'])
    selected_group_positive = grp_pn.get_group('positive')
    selected_group_negative = grp_pn.get_group('negative')
    
    for r in binarize_list:  # loops over every ROIs in atlas
        selected_roi_p = selected_group_positive[selected_group_positive['ROI'] == r]
        selected_roi_n = selected_group_negative[selected_group_negative['ROI'] == r]
        #select log10
        sig_p = selected_roi_p['P-value'].values[0]
        sig_n = selected_roi_n['P-value'].values[0]
        #select effect size
        r_p = selected_roi_p['RBC'].values[0]
        r_n = selected_roi_n['RBC'].values[0]
        p_out = str(r_p)
        n_out = str(r_n)   
        
        #if p value significant, replace lobule number with effect size
        if sig_p <= (.05/roi): #corrected -10 tasks in atlas######change
            seriesmap_p = seriesmap_p.replace(r, p_out)
        else:#if pvalue not significant, replace with zeros
            seriesmap_p = seriesmap_p.replace(r, 0)
        #select negative
        #x = (selected_roi_n['mean_hc'] >selected_roi_n['mean_c']).to_string(index=False)
        if sig_n <= (.05/roi):#corrected -10 tasks in atlas######change
            seriesmap_n = seriesmap_n.replace(r, n_out)
        else:
            seriesmap_n = seriesmap_n.replace(r, 0)
        df_r.append((p_out, n_out, r, m))

    c_Z_p = seriesmap_p.astype(float).values
    c_Z_n = seriesmap_n.astype(float).values
    file_name_n = str(m) + atlas_name+'_n_twosided_hc'  
    file_name_p = str(m)+atlas_name+'_p_twosided_hc' 

    wdir = "s_normative_model/s_suit_nm_5/effect_percentage_dev/"

    p_nii = pcn.dataio.fileio.save_nifti(
        c_Z_p, wdir+file_name_p+'_effect_allhc2.nii', examplenii=example_image, mask=mask_image)
    n_nii = pcn.dataio.fileio.save_nifti(
        c_Z_n, wdir+file_name_n+'_effect_allhc2.nii', examplenii=example_image, mask=mask_image)


    plt.gcf().set_dpi(300)
    funcdata_p = flatmap.vol_to_surf(wdir+file_name_p+'_effect_allhc2.nii',space='SUIT')
    flatmap.plot(data=funcdata_p, cmap='hot', threshold=[0,1], cscale=[0,.7],bordersize=1, bordercolor='w', new_figure=True,colorbar=True, render='matplotlib')
    plt.savefig(os.path.join(figure_dir,file_name_p +'_effect_allhc2_scale.png')) 
    plt.gcf().set_dpi(300)
    funcdata_n = flatmap.vol_to_surf(wdir+file_name_n+'_effect_allhc2.nii',space='SUIT')
    flatmap.plot(data=funcdata_n, cmap='hot', threshold=[0,1], cscale=[0,.7], bordersize=1, bordercolor='w', new_figure=True,colorbar=False, render='matplotlib')
    plt.savefig(os.path.join(figure_dir,file_name_n +'_effect_allhc2_scale.png'))


