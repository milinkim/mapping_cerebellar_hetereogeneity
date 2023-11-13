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
from scipy import stats

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
num_roi=10*2*5###change (roi x direciton x diagnosis) for multiple correction
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

#choose clinical group
disorder1=df_Z_te[df_Z_te['diagnosis']=='disorder1'] 
c=disorder1
c_name='disorder1'
c.set_index('ID', inplace=True)
del c['diagnosis']
del c['site']
c_t=c.T    

#concat symptom score/iq 
test_df=pd.read_csv('datasets/Composite_Phenotypic.csv')
test_df=test_df.dropna()
test_df['ID']=test_df['ID'].astype(str)
tests=['IQ', 'symptomscore']

for test in tests:
    r_p=[]
    for l in binarize_list: #loop through each lobules
            #binarize_t=binarize.T
            mul2=mul[0].replace(to_replace=0, method='ffill').values
            mul2=mul2.astype(float)
            binarize=mul2
            binarize[binarize!=l]=0
            binarize[binarize==l]=1
            binarize=binarize.T
            c_t_b=c_t * binarize[:,None]
            
            #get voxel number for each lobule
            n_lobules=np.unique(binarize, return_counts=True)
            n_lobules = [tup[1] for tup in n_lobules]
            v_lobules=n_lobules[1] #lobule number of voxel number 
            
            #get rid of rows with zero
            c_t_z=c_t_b[~(c_t_b == 0).all(axis=1)]
            
            #####get positive/negative deviations for clinical 
            df=[]
            for i in c_t_z.columns: #loop participant
                i_df= np.array(c_t_z[i]) 
                outlier_indices = np.where(np.logical_or(i_df > 3, i_df < -3))
                # Remove the outliers
                i_df2 = np.delete(i_df, outlier_indices)
                #get mean
                mean_df=i_df2.mean()
              #  i_df= np.array(c_t_z[i]) 
                p_outliers=0
                n_outliers=0
                n_subject=len(c_t_z.index)
                for j in i_df: #go through each Z scores 
                    if j >= 1.96:
                        p_outliers+=1
                    elif j <= -1.96:
                        n_outliers+=1
                #print(f'number of outliers for {i}:', n_outliers, 'percentage:',n_outliers/142978 )
                df.append((i, p_outliers, p_outliers/v_lobules, n_outliers, n_outliers/v_lobules, mean_df))
            
                #print(f'number of outliers for {i}:', n_outliers, 'percentage:',n_outliers/142978 )
            c_Z=pd.DataFrame(df, columns=('ID', 'num_p_outliers', 'p_p_outliers','num_n_outliers', 'p_n_outliers', 'mean_'))
            c_Z=c_Z.drop(['num_p_outliers', 'num_n_outliers', 'mean_'], axis=1)
            #c_Z['p_p_outliers_log']=np.log10(c_Z['p_p_outliers']+1)
            #c_Z['p_n_outliers_log']=np.log10(c_Z['p_n_outliers']+1)
                            
            #merge
            mm_merge=pd.merge(c_Z, test_df, on='ID') 
            mm_merge=mm_merge.set_index('ID')
            mm_merge = mm_merge[mm_merge[test].notna()]
            #mm_merge=mm_merge.apply(pd.to_numeric)
            mm_merge=mm_merge.loc[~mm_merge.index.duplicated(), :]
            
            x_col = test
            y_col = "p_p_outliers"
            y_col2="p_n_outliers"
            
            #percentage of positive
            r, p= stats.spearmanr(mm_merge[x_col],mm_merge[y_col])
            #percentage of negative 
            rr, pp= stats.spearmanr(mm_merge[x_col],mm_merge[y_col2])
            r_p.append((c_name,l,test, r,p,rr,pp,mm_merge.shape[0]))    
          
            
    r_p_df=pd.DataFrame(r_p, columns=('group', 'roi', 'test', 'p_p_correlation', 'p_p_pvalue','p_n_correlation', 'p_n_pvalue','number'))
    r_p_df.to_csv('/s_normative_model/correlation/'+c_name+'_correlation_'+atlas_name+'_'+test+'.csv')

    seriesmap=mul2
    seriesmap_p = seriesmap
    seriesmap_n = seriesmap
    
    df_r=[]
    for r in binarize_list:  # loops over every ROIs in atlas
        selected_roi_p = r_p_df['roi'] == r
        selected_roi_n = r_p_df['roi'] == r
        #select log10
        sig_p = selected_roi_p['p_p_pvalue'].values[0]
        sig_n = selected_roi_n['p_n_pvalue'].values[0]
        #select correlation size
        r_p = selected_roi_p['p_p_correlation'].values[0]
        r_n = selected_roi_n['p_n_correlation'].values[0]
        
        p_out = str(r_p)
        n_out = str(r_n)
         
        #s = (selected_roi_p['mean_hc'] >selected_roi_p['mean_c']).to_string(index=False)
        #if str2bool(s) and sig_p >= -1* np.log10(.05/10):
        #    seriesmap_p = seriesmap_p.replace(r, p_out)
        #else:
        #    seriesmap_p = seriesmap_p.replace(r, 0)
        
        #if p value significant, replace lobule number with effect size
        if sig_p <= 0.05/num_roi: #corrected above
            seriesmap_p = seriesmap_p.replace(r, p_out)
        else:#if pvalue not significant, replace with zeros
            seriesmap_p = seriesmap_p.replace(r, 0)
        #select negative
        #x = (selected_roi_n['mean_hc'] >selected_roi_n['mean_c']).to_string(index=False)
        if sig_n <= 0.05/num_roi:#corrected -10 tasks in atlas######change
            seriesmap_n = seriesmap_n.replace(r, n_out)
        else:
            seriesmap_n = seriesmap_n.replace(r, 0)
        df_r.append((p_out, n_out, r, c_name))

        #df_p=pd.concat([seriesmap_p, seriesmap_n], axis=1)

    c_Z_p = seriesmap_p.astype(float).values
    c_Z_n = seriesmap_n.astype(float).values
    file_name_n = c_name + '_'+atlas_name+'_'+test+'_correlation_per_n'  
    file_name_p = c_name+'_'+atlas_name+'_'+test+'_correlation_per_p' 

    wdir = "s_normative_model/correlation/"
    p_nii = pcn.dataio.fileio.save_nifti(
        c_Z_p, wdir+file_name_p+'_corr.nii', examplenii=example_image, mask=mask_image)
    n_nii = pcn.dataio.fileio.save_nifti(
        c_Z_n, wdir+file_name_n+'_corr.nii', examplenii=example_image, mask=mask_image)

#for voxelwise always save nii and then plot effect size
    plt.gcf().set_dpi(300)
    funcdata_p = flatmap.vol_to_surf(wdir+file_name_p+'_corr.nii',space='SUIT')
    flatmap.plot(data=funcdata_p, cmap='twilight', threshold=[0,1], cscale=[-.3,.3],bordersize=1, bordercolor='w', new_figure=True,colorbar=True, render='matplotlib')
    plt.savefig(os.path.join(figure_dir,file_name_p +'_corr.png')) 
    plt.gcf().set_dpi(300)
    funcdata_n = flatmap.vol_to_surf(wdir+file_name_n+'_corr.nii',space='SUIT')
    flatmap.plot(data=funcdata_n, cmap='twilight',threshold=[-1,0], cscale=[-.3,.3], bordersize=1, bordercolor='w', new_figure=True,colorbar=True, render='matplotlib')
    plt.savefig(os.path.join(figure_dir,file_name_n +'_corr.png'))

