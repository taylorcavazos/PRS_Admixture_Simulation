import numpy as np
import pandas as pd
import h5py
import os

import seaborn as sns
import matplotlib.pyplot as plt
import pingouin
from scipy import stats

from .emp_risk import _decrease_training_samples
"""
Create summary level information and plots for empirical and true PRS
"""

def output_all_summary(sim,m,h2,prefix,p,r2,
                        snp_weighting="ceu",snp_selection="ceu",
                        num2decrease=None):
    true_prs,emp_prs,anc,testing,train_cases,train_controls,labels = load_data(m,h2,r2,p,prefix,snp_selection,
                                                                        snp_weighting,num2decrease)
    sim = prefix.split('sim')[1].split('/')[0]
    if os.path.isfile(f"{prefix}summary/prs_corr_m_{m}_h2_{h2}_r2_{r2}_p_{p}"+\
                       f"_{snp_selection}_snps_{len(train_cases[snp_selection])}cases"+\
                        f"_{snp_weighting}_weights_{len(train_cases[snp_weighting])}cases_"+\
                       f"{sim}.txt"):
      print(f"\nSummary plots and data exist. If you would like to overwrite, remove "+\
            f"{prefix}summary/prs_corr_m_{m}_h2_{h2}_r2_{r2}_p_{p}"+\
            f"_{snp_selection}_snps_{len(train_cases[snp_selection])}cases"+\
            f"_{snp_weighting}_weights_{len(train_cases[snp_weighting])}cases_"+\
            f"{sim}.txt")
    
    else:
        anc_inds,summary = correlation(sim,true_prs,emp_prs,train_cases,train_controls,testing,anc,
                    m,h2,r2,p,snp_selection,snp_weighting,prefix,labels)

        plot_true_vs_empirical(prefix,true_prs,emp_prs,anc_inds,train_cases,snp_selection,snp_weighting,sim,
                                m,h2,r2,p)
        correlation_plot(summary,prefix,true_prs,emp_prs,anc_inds,train_cases,snp_selection,snp_weighting,sim,
                                m,h2,r2,p)

    return

def plot_true_vs_empirical(prefix,true_prs,emp_prs,anc_inds,train_cases,snp_selection,snp_weight,sim,
    m,h2,r2,p):
    labels = {"ceu":"CEU","yri":"YRI","low":"CEU <= 20%","mid":"80% > CEU > 20%","high":"CEU >= 80%"}
    colors = dict(zip(['ceu', 'high', 'mid', 'low', 'yri'],['#103c42', '#0b696a', '#069995','#8ac766','#ffe837']))
    titles = ["True PRS", "Empirical PRS"]

    fig,axes = plt.subplots(ncols=2,nrows=1,figsize=(20,10))
    for key in ['ceu', 'high', 'mid', 'low', 'yri']:
        for ind,prs in enumerate([true_prs,emp_prs]):
            sns.distplot(stats.zscore(prs)[anc_inds[key]],color=colors[key],label=labels[key],
                hist=False,kde=True,kde_kws = {'shade': True, 'linewidth': 3},ax=axes[ind])
            axes[ind].set_xlabel(titles[ind],fontsize=20)
            axes[ind].set_ylabel("Density",fontsize=20)
            axes[ind].set_xticklabels(axes[ind].get_xticks(),fontsize=20)
            axes[ind].set_yticklabels(np.round(axes[ind].get_yticks(),2),fontsize=20)

    fig.tight_layout(h_pad= 2,w_pad=0.5)
    sns.despine()
    plt.savefig(f"{prefix}summary/prs_density_m_{m}_h2_{h2}_r2_{r2}_p_{p}"+\
                    f"_{snp_selection}_snps_{len(train_cases[snp_selection])}cases"+\
                    f"_{snp_weight}_weights_{len(train_cases[snp_weight])}cases_"+\
                    f"{sim}.png",dpi=400)

    return
    
def correlation(sim,true_prs,emp_prs,train_cases,train_controls,testing,anc,
                m,h2,r2,p,snp_selection,snp_weight,prefix,labels):
    anc_inds = {}

    summary = pd.DataFrame(index=["vals"], columns=["train_ceu_corr","test_ceu_corr",
                                                    "train_yri_corr","test_yri_corr",
                                                    "test_admix_corr","test_admix_corr_par",
                                                    "admix_low_ceu_corr",
                                                    "admix_mid_ceu_corr","admix_high_ceu_corr"])
    for pop in ["ceu","yri","admix"]:
        if pop != "admix":
            train_true_prs = true_prs[np.append(train_cases[pop],train_controls[pop])]
            train_emp_prs = emp_prs[np.append(train_cases[pop],train_controls[pop])]
            summary.loc["vals",f"train_{pop}_corr"] = stats.pearsonr(train_true_prs,train_emp_prs)[0]
        
        test_true_prs = true_prs[testing[pop]]
        test_emp_prs = emp_prs[testing[pop]]
        if pop == "admix":
            pin_df = pd.DataFrame(np.array([test_true_prs,test_emp_prs,anc["Prop_CEU"].values]).transpose(),columns = ["true","emp","anc"])
            out_partial = pingouin.partial_corr(x="true",y="emp",covar="anc",data=pin_df)
            summary.loc["vals",f"test_{pop}_corr_par"] = out_partial["r"].values[0]
        summary.loc["vals",f"test_{pop}_corr"] = stats.pearsonr(test_true_prs,test_emp_prs)[0]

        anc_inds[pop] = testing[pop]

    for prop in [(0,0.2,"low"),(0.2,0.8,"mid"),(0.8,1,"high")]:
        prop_admix = anc[(anc["Prop_CEU"]>prop[0])&(anc["Prop_CEU"]<=prop[1])].index
        testing_prop_admix = testing["admix"][prop_admix]
        summary.loc["vals",f"admix_{prop[2]}_ceu_corr"] = stats.pearsonr(true_prs[testing_prop_admix],
                                                                          emp_prs[testing_prop_admix])[0]

        anc_inds[prop[2]] = testing["admix"][prop_admix]

    summary.to_csv(f"{prefix}summary/prs_corr_m_{m}_h2_{h2}_r2_{r2}_p_{p}"+\
                    f"_{snp_selection}_snps_{len(train_cases[snp_selection])}cases"+\
                    f"_{snp_weight}_weights_{len(train_cases[snp_weight])}cases_"+\
                    f"{sim}.txt",sep="\t")
    return anc_inds,summary


def correlation_plot(summary,prefix,true_prs,emp_prs,anc_inds,train_cases,snp_selection,snp_weight,sim,
                                m,h2,r2,p):
    labels = {"ceu":"CEU","yri":"YRI","low":"CEU <= 20%","mid":"80% > CEU > 20%","high":"CEU >= 80%"}
    colors = dict(zip(['ceu', 'high', 'mid', 'low', 'yri'],['#103c42', '#0b696a', '#069995','#8ac766','#ffe837']))
    corr_names = {"ceu":"test_ceu_corr","yri":"test_yri_corr","low":"admix_low_ceu_corr","mid":"admix_mid_ceu_corr","high":"admix_high_ceu_corr"}
    titles = ["True PRS", "Empirical PRS"]

    fig,axes = plt.subplots(ncols=5,nrows=1,figsize=(25,5),sharey=True)
    for ind,key in enumerate(['ceu', 'high', 'mid', 'low', 'yri']):
        axes[ind].scatter(stats.zscore(emp_prs)[anc_inds[key]],stats.zscore(true_prs)[anc_inds[key]],
                         color=colors[key])
        axes[ind].set_title(labels[key]+"\nPearson's rho = {}".format(np.round(summary.loc["vals",corr_names[key]],2)))
        axes[ind].set_ylabel("True PRS")
        axes[ind].set_xlabel("Emprirical PRS")
    fig.tight_layout(h_pad= 2,w_pad=0.5)
    sns.despine()
    plt.savefig(f"{prefix}summary/prs_corr_scat_m_{m}_h2_{h2}_r2_{r2}_p_{p}"+\
                    f"_{snp_selection}_snps_{len(train_cases[snp_selection])}cases"+\
                    f"_{snp_weight}_weights_{len(train_cases[snp_weight])}cases_"+\
                    f"{sim}.png",dpi=400)


def load_data(m,h2,r2,p,prefix,snp_selection,snp_weight,num2decrease):
    anc = pd.read_csv(f"{prefix}admixed_data/output/admix_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{snp_selection}_snps.prop.anc.PRS",
                        sep="\t")

    if num2decrease == None:
        f = h5py.File(f'{prefix}true_prs/prs_m_{m}_h2_{h2}.hdf5', 'r')
        train_cases = {"ceu":f["train_cases_ceu"][()],"yri":f["train_cases_yri"][()],
                       "meta":np.append(f["train_cases_ceu"][()],f["train_cases_yri"][()]),
                       "la":f["train_cases_yri"][()]}
        train_controls = {"ceu":f["train_controls_ceu"][()],"yri":f["train_controls_yri"][()],
                       "meta":np.append(f["train_controls_ceu"][()],f["train_controls_yri"][()]),
                       "la":f["train_controls_yri"][()]}
        labels_all = f["labels"][()]
        f.close()
    else:
        sub_yri_case,sub_yri_control = _decrease_training_samples(m,h2,"yri",num2decrease,prefix)
        f = h5py.File(f'{prefix}true_prs/prs_m_{m}_h2_{h2}.hdf5', 'r')
        train_cases = {"ceu":f["train_cases_ceu"][()],"yri":sub_yri_case,
                       "meta":np.append(f["train_cases_ceu"][()],sub_yri_case),
                       "la":sub_yri_case}
        train_controls = {"ceu":f["train_controls_ceu"][()],"yri":sub_yri_control,
                          "meta":np.append(f["train_controls_ceu"][()],sub_yri_control),
                          "la":sub_yri_control}
        labels_all = f["labels"][()]
        f.close() 

    all_testing = h5py.File(f"{prefix}true_prs/prs_m_{m}_h2_{h2}.hdf5","r")['test_data'][()]
    testing =  {"ceu":np.array([]),"yri":np.array([]),"admix":np.array([])}

    for ind in all_testing:
        label = labels_all[ind].decode("utf-8")
        if "ceu" in label: testing["ceu"] = np.append(testing["ceu"],[ind]).astype(int)
        elif "yri" in label: testing["yri"]= np.append(testing["yri"],[ind]).astype(int)
        else: testing["admix"]= np.append(testing["admix"],ind).astype(int)

    f.close()
    
    true_prs = h5py.File(f"{prefix}true_prs/prs_m_{m}_h2_{h2}.hdf5","r")['X'][()]
    emp_prs = h5py.File(f"{prefix}emp_prs/emp_prs_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{snp_selection}_snps_{len(train_cases[snp_selection])}cases"+\
                        f"_{snp_weight}_weights_{len(train_cases[snp_weight])}cases.hdf5","r")['X'][()]

    return true_prs,emp_prs,anc,testing,train_cases,train_controls,labels_all
