#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:11:03 2023

@author: p33-milink
"""
#plot deviation for roi

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:16:02 2022

@author: p33-milink
"""
import pandas as pd 
import glob
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
import nibabel as nb
import numpy as np
import pingouin as pg
from scipy.stats import mannwhitneyu
from numpy import mean,std
from statistics import mean, stdev
from math import sqrt

###########################load data and directories###########################

data_dir = '/s_aca_nm_5/data/'

cov_dir='/acapulco/covariates/'

df_tr=pd.read_csv(cov_dir+'bl_cov_tr.csv')
df_te=pd.read_csv(cov_dir+'bl_cov_te.csv')
df_clinical=pd.read_csv(cov_dir+'bl_cov_clinical.csv')
df_te_c=pd.read_csv(cov_dir+'bl_cov_te_clinical.csv')

#perform statistical test on extreme 
processing_dir= '/s_normative_model/s_aca_nm_5/clinical_model/'
figure_dir='/s_normative_model/figures/'

roi_ids= [
       'Corpus.Medullare', 'Left.Crus.I', 'Left.Crus.II', 'Left.I.III',
       'Left.IV', 'Left.IX', 'Left.V', 'Left.VI', 'Left.VIIB',
       'Left.VIIIA', 'Left.VIIIB', 'Left.X', 'Right.Crus.I',
       'Right.Crus.II', 'Right.I.III', 'Right.IX', 'Right.V',
       'Right.VI', 'Right.VIIB', 'Right.VIIIA', 'Right.VIIIB',
       'Right.X', 'Rigt.IV', 'Vermis.IX', 'Vermis.VI', 'Vermis.VII',
       'Vermis.VIII', 'Vermis.X'] 

#concat Z estimates for ROI
roi_Z=[]
for i in roi_ids:
    Z = pd.read_csv(processing_dir+ i+'/Z_estimate.txt', header=None, index_col=False)
    Z=Z.transpose()
    roi_Z.append(Z) 
df_roi= pd.concat(roi_Z)
df_roi.index=roi_ids
df_roi=df_roi.transpose()
df_roi=pd.merge(df_roi, df_te_c, left_index=True, right_index=True)
df_Z_te=df_roi.set_index('ID')


#group them into diagnosis df
num_clinical = 2 #multiple comparison 
hc=df_Z_te[df_Z_te['diagnosis']=='HC'] #hc
disorder1=df_Z_te[df_Z_te['diagnosis']=='disorder1'] 
disorder2=df_Z_te[df_Z_te['diagnosis']=='disorder2']

df=[]
groups=[hc, disorder1, disorder2]
groups_s=['hc', 'disorder1', 'disorder2']
#delete all columns except roi features
for d in groups:
    del d['diagnosis']
    del d['sex']
    del d['site']
    del d['machine']
    del d['model']
    del d['project']
    del d['age']

X1=pd.read_csv('/suit/t1_n4_mni_seg_post_inverse.txt', header=None)
X1[0] = X1[0].replace(12,"Corpus.Medullare")
X1[0] = X1[0].replace(33,"Left.I.III")
X1[0] = X1[0].replace(36,"Right.I.III")
X1[0] = X1[0].replace(43,"Left.IV")
X1[0] = X1[0].replace(46,"Rigt.IV")
X1[0] = X1[0].replace(53,"Left.V")
X1[0] = X1[0].replace(56,"Right.V")
X1[0] = X1[0].replace(60,"Vermis.VI")
X1[0] = X1[0].replace(63,"Left.VI")
X1[0] = X1[0].replace(66,"Right.VI")
X1[0] = X1[0].replace(70,"Vermis.VII")
X1[0] = X1[0].replace(73,"Left.Crus.I")
X1[0] = X1[0].replace(74,"Left.Crus.II")
X1[0] = X1[0].replace(75,"Left.VIIB")
X1[0] = X1[0].replace(76,"Right.Crus.I")
X1[0] = X1[0].replace(77,"Right.Crus.II")
X1[0] = X1[0].replace(78,"Right.VIIB")
X1[0] = X1[0].replace(80,"Vermis.VIII")
X1[0] = X1[0].replace(83,"Left.VIIIA")
X1[0] = X1[0].replace(84,"Left.VIIIB")
X1[0] = X1[0].replace(86,"Right.VIIIA")
X1[0] = X1[0].replace(87,"Right.VIIIB")
X1[0] = X1[0].replace(90,"Vermis.IX")
X1[0] = X1[0].replace(93,"Left.IX")
X1[0] = X1[0].replace(96,"Right.IX")
X1[0] = X1[0].replace(100,"Vermis.X")
X1[0] = X1[0].replace(103,"Left.X")
X1[0] = X1[0].replace(106,"Right.X")


############## percentage of extreme deviation for each groups ##############
df=[]
df_k=[]
seriesmap=X1[0]
for k, m in zip( groups, groups_s):
    for i in roi_ids:
        i_df= np.array(k[i]) #each brain region
        p_outliers=0
        n_outliers=0
        n_subject=len(k.index)
        for j in i_df: #go through each Z scores if above or below +/-1.96 
            if j >= 1.96:
                p_outliers+=1
            elif j <= -1.96:
                n_outliers+=1
        
        df.append((m, i, p_outliers, p_outliers/n_subject, n_outliers, n_outliers/n_subject))
    

    #print(f'number of outliers for {i}:', n_outliers, 'percentage:',n_outliers/142978 )
    df_k=pd.DataFrame(df, columns=('diagnosis','roi', 'num_p_outliers', 'p_p_outliers','num_n_outliers', 'p_n_outliers'))
    #df_k.to_csv()
    
wdir="/s_normative_model/s_aca_nm_5/percent_deviation/"
#load mask and example image and test
mask_image='/acapulco/output/t1/t1_bin.nii' #mask thresholded at 0.2
example_image='/acapulco/output/t1/t1_n4_mni_seg_post_inverse.nii'
seriesmap=X1[0] 
grp=df_k.groupby(['diagnosis'])

######################create nii file for each groups##########################
df_r=[]
for m in groups_s:
    selected_group = grp.get_group(m) #diagnosis group
    seriesmap_p=seriesmap
    seriesmap_n=seriesmap
    for r in roi_ids: #loops over every ROIs
        selected_roi=selected_group[selected_group['roi']==r]

        p_out=str(selected_roi['p_p_outliers'].values[0])
        n_out=str(selected_roi['p_n_outliers'].values[0])
        seriesmap_p=seriesmap_p.replace(r,p_out)
        seriesmap_n=seriesmap_n.replace(r,n_out)
        df_r.append((p_out, n_out, r,m))
                       
    c_Z_p=seriesmap_p.astype(float).values
    c_Z_n=seriesmap_n.astype(float).values
    file_name_n= str(m) +'_n_roi' #change name 
    file_name_p= str(m)+'_p_roi' 
    
    pcn.dataio.fileio.save_nifti(c_Z_p, wdir+file_name_p+'.nii', examplenii=example_image, mask=mask_image)
    pcn.dataio.fileio.save_nifti(c_Z_n, wdir+file_name_n+'.nii', examplenii=example_image, mask=mask_image)
    #positive deviation
    funcdata_p = flatmap.vol_to_surf(wdir+file_name_p+'.nii',space='SUIT')
    flatmap.plot(data=funcdata_p, cmap='hot', threshold=[0.00, 0.2],cscale=[0.00, 0.1], bordersize=.01, bordercolor='w',new_figure=True,colorbar=True, render='matplotlib')
    plt.savefig(os.path.join(figure_dir, file_name_p+'.png'))
    #negative deviaiton
    funcdata_n = flatmap.vol_to_surf(wdir+file_name_n+'.nii',space='SUIT')
    flatmap.plot(data=funcdata_n, cmap='hot', threshold=[0.00, 0.2],cscale=[0.00, 0.1], bordersize=.01, bordercolor='w',new_figure=True,colorbar=True, render='matplotlib')
    plt.savefig(os.path.join(figure_dir, file_name_n+'.png'))

df_i_df=pd.DataFrame(df_r, columns=('p_deviation', 'n_deviation','roi','clinical'))
df_i_df.to_csv('/s_normative_model/deviation_roi_per_lobules.csv')



#############perform statistical test on z-scores
result=[]
for g,s in zip(groups, groups_s): 
    #for each ROI
    for i in roi_ids:
        effect=pg.mwu(hc[i], g[i], alternative='two-sided')
        alpha = 0.05/(28*num_clinical)
        if effect['p-val'][0] >= alpha:
            result.append((i,s,effect['U-val'][0],effect['p-val'][0],effect.RBC[0],effect.CLES[0], round(hc[i].median(),3), round(g[i].median(),3), hc.shape[0], g.shape[0]))
        else:
            result.append((i,s,effect['U-val'][0],effect['p-val'][0],effect.RBC[0],effect.CLES[0], round(hc[i].median(),3), round(g[i].median(),3), hc.shape[0], g.shape[0]))

 
df_df=pd.DataFrame(result, columns=('roi','diagnosis','stat_uvalue', 'pvalue', 'rbc','cles','median_hc','median_c', 'hc_n','c_n'))
df_df=df_df[df_df.diagnosis != 'hc']
df_df.to_csv('/figures/nonparametric_test_lobules.csv')

#plot rbc to cerebellum 
seriesmap=X1[0] 
grp=df_df.groupby(['diagnosis'])
df_r=[]
groups_ss=[ 'asd' ,'ad', 'mci', 'bd', 'scz'] #greoup without hc

for m in groups_ss:
    selected_group = grp.get_group(m) #diagnosis group
    seriesmap_z=seriesmap
    for r in roi_ids: #loops over every ROIs
        selected_roi=selected_group[selected_group['roi']==r]
        rbc=selected_roi['rbc'].values[0]
        sig_p=selected_roi['pvalue'].values[0]
        if sig_p <= (.05/(28*num_clinical*2)): #birectional
            seriesmap_z = seriesmap_z.replace(r, rbc)
        else:#if pvalue not significant, replace with zeros
            seriesmap_z = seriesmap_z.replace(r, 0)
    c_Z_z = seriesmap_z.astype(float).values
    file_name_z = str(m) + 'lobules'+'_z_twosided_hc'  
    wdir = "/s_normative_model/s_aca_nm_5/percent_deviation/"
    z_nii = pcn.dataio.fileio.save_nifti(c_Z_z, wdir+file_name_z+'_effect.nii', examplenii=example_image, mask=mask_image)
    plt.gcf().set_dpi(300)
    funcdata_z = flatmap.vol_to_surf(wdir+file_name_z+'_effect.nii',space='SUIT')
    flatmap.plot(data=funcdata_z, cmap='hot_r', threshold=[-0.3,0], cscale=[-0.3,0],bordersize=1, bordercolor='w', new_figure=True,colorbar=True, render='matplotlib')
    plt.savefig(os.path.join(figure_dir,file_name_z +'_effect.png')) 
   
    

