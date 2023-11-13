"""
Created on Wed Jan 11 10:11:03 2023

@author: Thomas Wolfers and Milin Kim
"""

from __future__ import with_statement
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pingouin as pg
from pcntoolkit.normative import estimate, predict, evaluate
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix,calibration_descriptives
import os
import sys
import numpy as np
from scipy import stats
from subprocess import call
from scipy.stats import genextreme, norm
from six import with_metaclass
from abc import ABCMeta, abstractmethod
import pickle
import pandas as pd
import bspline
from bspline import splinelab
from sklearn.datasets import make_regression
import pymc3 as pm
from io import StringIO
import subprocess
import re
from sklearn.metrics import roc_auc_score
import scipy.special as spp
###################################CONFIG #####################################
#set directories
root_dir = '/s_normative_model/'
blr_dir=os.path.join(root_dir,'blr')
dummy_dir = os.path.join(root_dir,'s_suit_nm','clinical_model','s_dummy')
out_dir = os.path.join(root_dir,'clinical_model')
os.makedirs(os.path.join(out_dir), exist_ok=True)
os.makedirs(os.path.join(dummy_dir), exist_ok=True)
#read in one of the splitted SUIT output
df_tr=pd.read_pickle('/s_suit_split_chunk/suit_chunk_1_tr.pkl')
df_te=pd.read_pickle('/s_suit_split_chunk/suit_chunk_1_te_clinical.pkl')

#set covariates
cols_cov = ['age','sex']
#site ID
site_ids =  sorted(set(df_tr['site'].to_list())) 

#list of all column of voxels [0-142977]
with open(os.path.join(root_dir,'idp_ids.txt')) as f:
    idp_ids = f.read().splitlines()

# run switches
show_plot = True
force_refit = True

warp =  'WarpSinArcsinh'   # 'WarpBoxCox', 'WarpSinArcsinh'  or None
sex = 0 # 1 = male 0 = female
if sex == 0: 
    clr = 'red';
else:
    clr = 'blue'

# age min and max
xmin = 0 
xmax = 85

# create dummy data for visualisation
xx = np.arange(xmin,xmax,0.5)
if len(cols_cov) == 1:
    print('fitting sex specific model')
    X0_dummy = np.zeros((len(xx), 1))
    X0_dummy[:,0] = xx
    df_tr = df_tr.loc[df_tr['sex'] == sex]
    df_te = df_te.loc[df_te['sex'] == sex]
else:
    X0_dummy = np.zeros((len(xx), 2))
    X0_dummy[:,0] = xx
    X0_dummy[:,1] = sex
print('configuring dummy data for each all',len(site_ids), 'sites')  
for sid, site in enumerate(site_ids): 
    site_dummy = [site] * len(xx)
    X_dummy = create_design_matrix(X0_dummy, xmin=xmin, xmax=xmax, site_ids=site_dummy, all_sites = site_ids)
    np.savetxt(os.path.join(dummy_dir,'cov_bspline_dummy_' + site + '.txt'), X_dummy)
print('configuring dummy data for mean')
X_dummy = create_design_matrix(X0_dummy, xmin=xmin, xmax=xmax, site_ids=None, all_sites = site_ids)
X_dummy_df=pd.DataFrame(X_dummy)
np.savetxt(os.path.join(dummy_dir,'cov_bspline_dummy_mean.txt'), X_dummy)
X_dummy_df.to_csv('/cluster/s_dummy/cov_bspline_dummy_mean.csv', index=False)

#create b_spline
X_tr = create_design_matrix(df_tr[cols_cov], site_ids = df_tr['site'],
                                basis = 'bspline', xmin = xmin, xmax = xmax)
X_tr_df=pd.DataFrame(X_tr)
X_tr_df.to_csv('/cluster/s_dummy/cov_bspline_tr.csv', index=False)

#change this
X_te = create_design_matrix(df_te[cols_cov], site_ids = df_te['site'], all_sites=site_ids,
                                basis = 'bspline', xmin = xmin, xmax = xmax)
X_te_df=pd.DataFrame(X_te)
X_te_df.to_csv('/cluster/s_dummy/cov_bspline_te.csv', index=False)

# configure and save the covariates
cov_file_tr = os.path.join(dummy_dir, 'cov_bspline_tr.txt')
cov_file_te = os.path.join(dummy_dir, 'cov_bspline_te.txt')
np.savetxt(cov_file_tr, X_tr)
np.savetxt(cov_file_te, X_te)
    

#for split-half
#sort by site, age and sex -take every second: hc -> training/test 
#then clinical groups include with the half of the hc-> test 
chunk=sys.argv[1]

#read Training splitted chunks of SPLIT
df_tr= pd.read_pickle('/s_normative_model/s_suit_split_chunk/suit_chunk_'+str(chunk)+'_tr.pkl')
df_tr.columns= df_tr.columns.astype(str) #convert column to strings
df_tr=df_tr.set_index('ID')
del df_tr['diagnosis']

#read Testing splitted chunks of SPLIT
df_te= pd.read_pickle('/s_normative_model/s_suit_split_chunk/suit_chunk_'+str(chunk)+'_te_clinical.pkl')
df_te.columns= df_te.columns.astype(str)
df_te=df_te.set_index('ID')
del df_te['diagnosis']

#read s_dummy
dummy_dir = os.path.join(root_dir,'clinical_model', 's_dummy')
out_dir = os.path.join(root_dir,'clinical_model',str(chunk))
os.makedirs(os.path.join(out_dir), exist_ok=True)


################################### RUN #######################################

xx = np.arange(xmin,xmax,0.5)


blr_metrics = pd.DataFrame(columns = ['eid', 'NLL', 'EV', 'MSLL', 'BIC','Skew','Kurtosis','RMSE', 'SMSE', 'Rho'])
for nummer, idp in enumerate(idp_ids): 
    print(nummer)
    print('Running IDP:', idp)
    idp_dir = os.path.join(out_dir, str(idp))
    
    # set output dir 
    os.makedirs(os.path.join(idp_dir), exist_ok=True)
    os.chdir(idp_dir)
    
    # configure and save the responses
    # configure and save the responses
    y_tr = df_tr[idp].to_numpy() 
    y_te = df_te[idp].to_numpy()

    #impute bad subjects with std5
    mean_tr=np.mean(y_tr, axis=0)
    std_tr=np.std(y_tr, axis=0)
    std=4   # standard deviation
    y_tr_upper= mean_tr + std * std_tr
    y_tr_lower= mean_tr - std * std_tr
    #replace each value by the mean_tr that falls outside range 5x std 
    y_tr=np.where(y_tr < y_tr_lower, mean_tr, y_tr)
    y_te=np.where(y_te > y_tr_upper, mean_tr, y_te)
    #replace na
    y_tr_series=pd.Series(y_tr)
    y_te_series=pd.Series(y_te)
    y_tr=np.where(y_tr_series.isnull(), mean_tr, y_tr)
    y_te=np.where(y_te_series.isnull(), mean_tr, y_te)

    
    resp_file_tr = os.path.join(idp_dir, 'resp_tr.txt')
    resp_file_te = os.path.join(idp_dir, 'resp_te.txt') 
    np.savetxt(resp_file_tr, y_tr)
    np.savetxt(resp_file_te, y_te)
    
    y_tr = y_tr[:, np.newaxis]  
    y_te = y_te[:, np.newaxis]
    
    # configure and save the covariates
    #X_tr = create_design_matrix(df_tr[cols_cov].loc[nz_tr], site_ids = df_tr['site'].loc[nz_tr],
         #                       basis = 'bspline', xmin = xmin, xmax = xmax)
    #X_te = create_design_matrix(df_te[cols_cov].loc[nz_te], site_ids = df_te['site'].loc[nz_te], all_sites=site_ids,
          #                      basis = 'bspline', xmin = xmin, xmax = xmax)

    # configure and save the covariates
    #cov_file_tr = os.path.join(idp_dir, 'cov_bspline_tr.txt')
    #cov_file_te = os.path.join(idp_dir, 'cov_bspline_te.txt')
    #np.savetxt(cov_file_tr, X_tr)
    #np.savetxt(cov_file_te, X_te)
    X_tr=pd.read_csv(os.path.join(dummy_dir,'cov_bspline_tr.csv'))
    X_tr=X_tr.to_numpy()
    X_te=pd.read_csv(os.path.join(dummy_dir,'cov_bspline_te.csv'))
    X_te=X_te.to_numpy()
    cov_file = os.path.join(dummy_dir, 'cov_bspline')
    cov_file_te = cov_file + '_te.txt'
    cov_file_tr = cov_file + '_tr.txt'
    #np.savetxt(cov_file_tr, X_tr)
    #np.savetxt(cov_file_te, X_te)
    

    if not force_refit and os.path.exists(os.path.join(idp_dir, 'Models', 'NM_0_0_estimate.pkl')):
        print('Making predictions using a pre-existing model')
        
        # Make prdictsion with test data
        yhat_te, s2_te, Z = predict(cov_file_te, alg='blr', 
                                    respfile=resp_file_te, 
                                    model_path=os.path.join(idp_dir,'Models'), 
                                    outputsuffix='_predict')
        estimated = False
    else:
        estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te, 
                 testcov=cov_file_te, alg='blr', optimizer = 'l-bfgs-b', 
                 savemodel=True, warp=warp, warp_reparam=True) 
        yhat_te = np.loadtxt(os.path.join(idp_dir, 'yhat_estimate.txt'))
        s2_te = np.loadtxt(os.path.join(idp_dir, 'ys2_estimate.txt'))
        yhat_te = yhat_te[:, np.newaxis]
        s2_te = s2_te[:, np.newaxis]
        estimated = True

    
    # set up the dummy covariates for the dummy data
    print('Making predictions with dummy covariates (for visualisation)')
    cov_file_dummy = os.path.join(dummy_dir, 'cov_bspline_dummy')
    cov_file_dummy = cov_file_dummy + '_mean.txt'
    # make dummy predictions
    yhat, s2 = predict(cov_file_dummy, alg='blr', respfile=None, 
                       model_path=os.path.join(idp_dir,'Models'), 
                       outputsuffix='_dummy')
    
    with open(os.path.join(idp_dir,'Models', 'NM_0_0_fit.pkl'), 'rb') as handle:
        nm = pickle.load(handle) 
    
    # compute error metrics
    if warp is None:
        # compute evaluation metrics
        metrics = evaluate(y_te, yhat_te)  
        
        # compute MSLL manually as a sanity check
        y_tr_mean = np.array( [[np.mean(y_tr)]] )
        y_tr_var = np.array( [[np.var(y_tr)]] )
        MSLL = compute_MSLL(y_te, yhat_te, s2_te, y_tr_mean, y_tr_var)         
     
    else:
        warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
        W = nm.blr.warp
        
        # warp and plot dummy predictions
        med, pr_int = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param)
        
        # warp predictions
        med_te = W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), warp_param)[0]
        med_te = med_te[:, np.newaxis]
       
        # evaluation metrics
        metrics = evaluate(y_te, med_te)
        
        # compute MSLL manually
        y_te_w = W.f(y_te, warp_param)
        y_tr_w = W.f(y_tr, warp_param)
        y_tr_mean = np.array( [[np.mean(y_tr_w)]] )
        y_tr_var = np.array( [[np.var(y_tr_w)]] )
        MSLL = compute_MSLL(y_te_w, yhat_te, s2_te, y_tr_mean, y_tr_var)     
  
    # plot the data points
    y_te_rescaled_all = np.zeros_like(y_te)
    for sid, site in enumerate(site_ids):
                
        # plot the true test data points
        if len(cols_cov) == 1:
            # sex-specific model
            idx = np.where(X_te[:,sid+len(cols_cov)+1] !=0)
        else:
            idx = np.where(np.bitwise_and(X_te[:,2] == sex, X_te[:,sid+len(cols_cov)+1] !=0)) #look into 
        if len(idx[0]) == 0:
            print('No data for site', sid, site, 'skipping...')
            continue
        else:
            X_dummy=pd.read_csv(os.path.join(dummy_dir,'cov_bspline_dummy_mean.csv'))
            X_dummy=X_dummy.to_numpy()
            idx_dummy = np.bitwise_and(X_dummy[:,1] > X_te[idx,1].min(), X_dummy[:,1] < X_te[idx,1].max())
        
        # adjust the intercept
        if warp is None:
            y_te_rescaled = y_te[idx] - np.median(y_te[idx]) + np.median(yhat[idx_dummy])
        else:            
            y_te_rescaled = y_te[idx] - np.median(y_te[idx]) + np.median(med[idx_dummy])
        if show_plot:
            plt.scatter(X_te[idx,1], y_te_rescaled, s=4, color=clr, alpha = 0.05)   
        
        y_te_rescaled_all[idx] = y_te_rescaled

    # plot the centiles
    if warp is None:
        if show_plot:
            plt.plot(xx, yhat, color = clr)
            plt.fill_between(xx, np.squeeze(yhat-0.67*np.sqrt(s2)), 
                             np.squeeze(yhat+0.67*np.sqrt(s2)), 
                             color=clr, alpha = 0.1)
            plt.fill_between(xx, np.squeeze(yhat-1.64*np.sqrt(s2)), 
                             np.squeeze(yhat+1.64*np.sqrt(s2)), 
                             color=clr, alpha = 0.1)
            plt.fill_between(xx, np.squeeze(yhat-2.33*np.sqrt(s2)), 
                             np.squeeze(yhat+2.32*np.sqrt(s2)), 
                             color=clr, alpha = 0.1)
            plt.plot(xx, np.squeeze(yhat-0.67*np.sqrt(s2)),color=clr, linewidth=0.5)
            plt.plot(xx, np.squeeze(yhat+0.67*np.sqrt(s2)),color=clr, linewidth=0.5)
            plt.plot(xx, np.squeeze(yhat-1.64*np.sqrt(s2)),color=clr, linewidth=0.5)
            plt.plot(xx, np.squeeze(yhat+1.64*np.sqrt(s2)),color=clr, linewidth=0.5)
            plt.plot(xx, np.squeeze(yhat-2.33*np.sqrt(s2)),color=clr, linewidth=0.5)
            plt.plot(xx, np.squeeze(yhat+2.32*np.sqrt(s2)),color=clr, linewidth=0.5)
    else:
        warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
        W = nm.blr.warp
        
        # warp and plot dummy predictions
        med, pr_int = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param)
        
        beta, junk1, junk2 = nm.blr._parse_hyps(nm.blr.hyp, X_dummy)
        s2n = 1/beta
        s2s = s2-s2n
        
        # plot the centiles
        if show_plot: 
            plt.plot(xx, med, clr)
            # fill the gaps in between the centiles
            junk, pr_int25 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.25,0.75])
            junk, pr_int95 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.05,0.95])
            junk, pr_int99 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.01,0.99])
            
            plt.fill_between(xx, pr_int25[:,0], pr_int25[:,1], alpha = 0.1,color=clr)
            plt.fill_between(xx, pr_int95[:,0], pr_int95[:,1], alpha = 0.1,color=clr)
            plt.fill_between(xx, pr_int99[:,0], pr_int99[:,1], alpha = 0.1,color=clr)
            
            # make the width of each line proportional to the epistemic uncertainty
            junk, pr_int25l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.25,0.75])
            junk, pr_int95l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.05,0.95])
            junk, pr_int99l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.01,0.99])
            junk, pr_int25u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.25,0.75])
            junk, pr_int95u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.05,0.95])
            junk, pr_int99u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.01,0.99])    
            plt.fill_between(xx, pr_int25l[:,0], pr_int25u[:,0], alpha = 0.3,color=clr)
            plt.fill_between(xx, pr_int95l[:,0], pr_int95u[:,0], alpha = 0.3,color=clr)
            plt.fill_between(xx, pr_int99l[:,0], pr_int99u[:,0], alpha = 0.3,color=clr)
            plt.fill_between(xx, pr_int25l[:,1], pr_int25u[:,1], alpha = 0.3,color=clr)
            plt.fill_between(xx, pr_int95l[:,1], pr_int95u[:,1], alpha = 0.3,color=clr)
            plt.fill_between(xx, pr_int99l[:,1], pr_int99u[:,1], alpha = 0.3,color=clr)

            # plot ceitle lines
            plt.plot(xx, pr_int25[:,0],color=clr, linewidth=0.5)
            plt.plot(xx, pr_int25[:,1],color=clr, linewidth=0.5)
            plt.plot(xx, pr_int95[:,0],color=clr, linewidth=0.5)
            plt.plot(xx, pr_int95[:,1],color=clr, linewidth=0.5)
            plt.plot(xx, pr_int99[:,0],color=clr, linewidth=0.5)
            plt.plot(xx, pr_int99[:,1],color=clr, linewidth=0.5)
            

    if show_plot:
        plt.xlabel('Age')
        plt.ylabel(idp) 
        plt.title(idp)
        plt.xlim((0,90))
        #plt.ylim((350000,990000)) #how to make it the same for both sex?
        #plt.savefig(os.path.join(idp_dir, 'centiles_' + str(sex)),  bbox_inches='tight')
        plt.show()
     
    BIC = len(nm.blr.hyp) * np.log(y_tr.shape[0]) + 2 * nm.neg_log_lik
    
    # print -log(likelihood)
    print('NLL =', nm.neg_log_lik)
    print('BIC =', BIC)
    print('EV = ', metrics['EXPV'])
    print('MSLL = ', MSLL) 
    
    if estimated:
        Z = np.loadtxt(os.path.join(idp_dir, 'Z_estimate.txt'))
    else:
        Z = np.loadtxt(os.path.join(idp_dir, 'Z_predict.txt'))
    [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(Z)
    
    
    RMSE=np.loadtxt(os.path.join(idp_dir, 'RMSE_estimate.txt'))
    SMSE=np.loadtxt(os.path.join(idp_dir, 'SMSE_estimate.txt'))
    Rho= np.loadtxt(os.path.join(idp_dir, 'Rho_estimate.txt'))
    
    
    blr_metrics.loc[len(blr_metrics)] = [idp, nm.neg_log_lik, 
                                         metrics['EXPV'][0], MSLL[0], BIC,
                                         skew, kurtosis, RMSE, SMSE, Rho]

    if show_plot:
    #     plt.figure()
    #     plt.hist(Z, bins = 100, label = 'skew = ' + str(round(skew,3)) + ' kurtosis = ' + str(round(kurtosis,3)))
    #     plt.title('Z_warp ' + idp)
    #     plt.legend()
    #     plt.savefig(os.path.join(idp_dir,'Z_hist'),  bbox_inches='tight')
    #     plt.show()
    
        plt.figure()
        #sm.qqplot(Z, line = '45')
        pg.qqplot(Z, dist='norm', confidence=0.95)
        #plt.savefig(os.path.join(idp_dir, 'Z_qq'+str(sex)),  bbox_inches='tight')
        plt.show()

blr_metrics.to_csv(os.path.join(blr_dir, str(chunk)+'_blr_metrics.csv'))

