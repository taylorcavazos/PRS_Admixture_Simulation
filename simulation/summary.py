import numpy as np
import pandas as pd
import h5py
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from .emp_risk import _decrease_training_samples

"""
Create summary level information and plots for empirical and true PRS
"""

def output_all_summary(sim,m,h2,prefix,p,r2,
                        snp_weighting="ceu",snp_selection="ceu",
                        num2decrease=None):
    """
    Function to calculate correlation between true and empirical PRS

    Parameters
    ----------
    sim : int
    	simulation identifier
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    prefix : str
        Output file path
    p : float
        P-value to use for summary statistic thresholding
    r2 : float
        LD r2 to use for summary statistic clumping
    snp_weighting : str, optional
    	population used for PRS variant weighting
    snp_selection : str, optional
    	population used for PRS variant selection
    num2decrease: int, optional
        Number of non-European samples to use in GWAS
    """
    # load true and empirical prs, global ancestry proportionals for admixed individuals,
    # training and testing samples and sample IDs 
    true_prs,emp_prs,anc,testing,train_cases,train_controls,labels = load_data(m,h2,r2,p,prefix,
    																		   snp_selection,
                                                                        	   snp_weighting,
                                                                        	   num2decrease)
    # extract simulation identifier
    sim = prefix.split('sim')[1].split('/')[0]
    # check if summary data exists
    if os.path.isfile(f"{prefix}summary/prs_corr_m_{m}_h2_{h2}_r2_{r2}_p_{p}"+\
                       f"_{snp_selection}_snps_{len(train_cases[snp_selection])}cases"+\
                        f"_{snp_weighting}_weights_{len(train_cases[snp_weighting])}cases_"+\
                       f"{sim}.txt"):
      # if summary data exists, ouptut warning
      print(f"\nSummary plots and data exist. If you would like to overwrite, remove "+\
            f"{prefix}summary/prs_corr_m_{m}_h2_{h2}_r2_{r2}_p_{p}"+\
            f"_{snp_selection}_snps_{len(train_cases[snp_selection])}cases"+\
            f"_{snp_weighting}_weights_{len(train_cases[snp_weighting])}cases_"+\
            f"{sim}.txt")
    
    else: # if summary data does not exist create plots and correlations
    	# calculate correlation between true and empirical PRS for each ancestry group
        anc_inds,summary = correlation(sim,true_prs,emp_prs,train_cases,train_controls,
        							   testing,anc,m,h2,r2,p,snp_selection,snp_weighting,
        							   prefix,labels)
        # plot true and empirical prs distributions broken up by ancestry
        plot_true_vs_empirical(prefix,true_prs,emp_prs,anc_inds,train_cases,snp_selection,
        					   snp_weighting,sim,m,h2,r2,p)
        # plot true vs empirical prs scatterplot for each ancestry group
        correlation_plot(summary,prefix,true_prs,emp_prs,anc_inds,train_cases,snp_selection,
        				 snp_weighting,sim,m,h2,r2,p)

    return

def plot_true_vs_empirical(prefix,true_prs,emp_prs,anc_inds,train_cases,snp_selection
						  ,snp_weight,sim,m,h2,r2,p):
    """
    Function to plot true and empirical PRS by ancestry

    Parameters
    ---------
    prefix : str
        Output file path
    true_prs : np.array
    	True PRS for all samples
    emp_prs : np.array
    	GWAS estimated PRS for all samples
    anc_inds : dict
    	ancestry population and indices in testing data
    train_cases : np.array
    	training cases used to create empirical PRS
    snp_selection : str
    	population used for PRS variant selection
    snp_weight : str
    	population used for PRS variant weighting
    sim : int
    	simulation identifier
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    r2 : float
        LD r2 to use for summary statistic clumping
    p : float
        P-value to use for summary statistic thresholding
    """
    # distribution labels
    labels = {"ceu":"CEU","yri":"YRI","low":"CEU <= 20%","mid":"80% > CEU > 20%","high":"CEU >= 80%"}
    # colors for each ancestry group
    colors = dict(zip(['ceu', 'high', 'mid', 'low', 'yri'],['#103c42', '#0b696a', '#069995','#8ac766','#ffe837']))
    # x and y labels
    titles = ["True PRS", "Empirical PRS"]
    # create and save figure
    fig,axes = plt.subplots(ncols=2,nrows=1,figsize=(20,10))
    for key in ['ceu', 'high', 'mid', 'low', 'yri']:
        for ind,prs in enumerate([true_prs,emp_prs]):
            sns.histplot(stats.zscore(prs)[anc_inds[key]],color=colors[key],label=labels[key],
                ax=axes[ind],kde=True)
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
    """
    Function to calculate correlation between true and empirical PRS

    Parameters
    ----------
    sim : int
    	simulation identifier
    true_prs : np.array
    	True PRS for all samples
    emp_prs : np.array
    	GWAS estimated PRS for all samples
    train_cases : np.array
    	training cases used to create empirical PRS
    train_controls : np.array
    	training controls used to create empirical PRS
    testing : np.array
    	independent testing data
    anc : pd.DataFrame
    	global ancestry for simulated admixed individuals
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    r2 : float
        LD r2 to use for summary statistic clumping
    p : float
        P-value to use for summary statistic thresholding
    snp_selection : str
    	population used for PRS variant selection
    snp_weight : str
    	population used for PRS variant weighting
    prefix : str
        Output file path
    labels : np.array
        sample IDs

    Returns
    -------
    dict
		ancestry population and indices in testing data
    pd.DataFrame
    	correlations between true and empirical PRS for European,
    	African, and admixed individuals
    """
    anc_inds = {} # initialize dictionary of ancestry populations and 
    # initialize correlation dataframe
    # true and empirical prs correlations reported for training and testing
    # Europeans, Africans, and admixed individuals broken down by nancestry
    summary = pd.DataFrame(index=["vals"], columns=["train_ceu_corr","test_ceu_corr",
                                                    "train_yri_corr","test_yri_corr",
                                                    "test_admix_corr",
                                                    "admix_low_ceu_corr",
                                                    "admix_mid_ceu_corr","admix_high_ceu_corr",
													"train_ceu_p","test_ceu_p",
                                                    "train_yri_p","test_yri_p",
                                                    "test_admix_p",
                                                    "admix_low_ceu_p",
                                                    "admix_mid_ceu_p","admix_high_ceu_p"])
    # for each population
    for pop in ["ceu","yri","admix"]:
        if pop != "admix":
        	# extract true and empirical PRS for population training data
            train_true_prs = true_prs[np.append(train_cases[pop],train_controls[pop])]
            train_emp_prs = emp_prs[np.append(train_cases[pop],train_controls[pop])]
            # calculate pearson's correlation for training
            summary.loc["vals",f"train_{pop}_corr"] = stats.pearsonr(train_true_prs,train_emp_prs)[0]
        # extract true and empirical PRS for population testing samples
        test_true_prs = true_prs[testing[pop]]
        test_emp_prs = emp_prs[testing[pop]]
        # calculate pearson's correlation and p-value for testing
        summary.loc["vals",f"test_{pop}_corr"] = stats.pearsonr(test_true_prs,test_emp_prs)[0]
        summary.loc["vals",f"test_{pop}_p"] = stats.pearsonr(test_true_prs,test_emp_prs)[1]
        # indices for testing population
        anc_inds[pop] = testing[pop]
    # break down admixed individuals by ancestry proportions
    for prop in [(0,0.2,"low"),(0.2,0.8,"mid"),(0.8,1,"high")]:
    	# subset admixed individuals by ancestry proportion
        prop_admix = anc[(anc["Prop_CEU"]>prop[0])&(anc["Prop_CEU"]<prop[1])].index
        # extract testing samples
        testing_prop_admix = testing["admix"][prop_admix]
        # calculate pearson's correlation for testing with admix proportion
        summary.loc["vals",f"admix_{prop[2]}_ceu_corr"] = stats.pearsonr(true_prs[testing_prop_admix],
                                                                          emp_prs[testing_prop_admix])[0]
        # calculate pearson's correlation for testing with admix proportion
        summary.loc["vals",f"admix_{prop[2]}_ceu_p"] = stats.pearsonr(true_prs[testing_prop_admix],
                                                                          emp_prs[testing_prop_admix])[1]
        # indices for testing population
        anc_inds[prop[2]] = testing["admix"][prop_admix]
    # write correlations
    summary.to_csv(f"{prefix}summary/prs_corr_m_{m}_h2_{h2}_r2_{r2}_p_{p}"+\
                    f"_{snp_selection}_snps_{len(train_cases[snp_selection])}cases"+\
                    f"_{snp_weight}_weights_{len(train_cases[snp_weight])}cases_"+\
                    f"{sim}.txt",sep="\t")
    return anc_inds,summary


def correlation_plot(summary,prefix,true_prs,emp_prs,anc_inds,
					 train_cases,snp_selection,snp_weight,sim,m,h2,r2,p):
    """
    Function to make scatterplot to show correlation between
    true and empirical PRS

    Parameters
    ----------
    prefix : str
        Output file path
    true_prs : np.array
    	True PRS for all samples
    emp_prs : np.array
    	GWAS estimated PRS for all samples
    anc_inds : dict
    	indices of testing samples broken down by ancestry
    train_cases : np.array
    	training cases used to create empirical PRS
    snp_selection : str
    	population used for PRS variant selection
    snp_weight : str
    	population used for PRS variant weighting
    sim : int
    	simulation identifier
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    r2 : float
        LD r2 to use for summary statistic clumping
    p : float
        P-value to use for summary statistic thresholding
    """
    # labels for plot
    labels = {"ceu":"CEU","yri":"YRI","low":"CEU <= 20%","mid":"80% > CEU > 20%","high":"CEU >= 80%"}
    # colors for plot by ancestry
    colors = dict(zip(['ceu', 'high', 'mid', 'low', 'yri'],['#103c42', '#0b696a', '#069995','#8ac766','#ffe837']))
    # columns in correlation summary dataframe
    corr_names = {"ceu":"test_ceu_corr","yri":"test_yri_corr","low":"admix_low_ceu_corr",
    			  "mid":"admix_mid_ceu_corr","high":"admix_high_ceu_corr"}
   	# titles for plot
    titles = ["True PRS", "Empirical PRS"]
    # create and save figure
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
    """
    Load true PRS, empirical PRS, global ancestry, and training
    and testing samples

    Parameters
    ----------
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    r2 : float
        LD r2 to use for summary statistic clumping
    p : float
        P-value to use for summary statistic thresholding
    prefix : str
        Output file path
    num2decrease: int
        Number of non-European samples to use in GWAS

    Returns
    -------
    np.array
    	true polygenic risk score
    np.array
    	empirical polygenic risk score
    pd.DataFrame
    	global ancestry admixed individuals
    np.array
    	testing samples
    np.array
    	training samples that are cases
    np.array
    	training samples that are controls
    np.array
    	sample IDs
    """
    # load global ancestry proportions for admixed individuals
    anc = pd.read_csv(f"{prefix}admixed_data/output/admix_afr_amer.prop.anc",sep="\t")
    # check if number of African samples is decreased
    if num2decrease == None: # if not load full dataset
        f = h5py.File(f'{prefix}true_prs/prs_m_{m}_h2_{h2}.hdf5', 'r')
        train_cases = {"ceu":f["train_cases_ceu"][()],"yri":f["train_cases_yri"][()],
                       "meta":np.append(f["train_cases_ceu"][()],f["train_cases_yri"][()]),
                       "la":f["train_cases_yri"][()]}
        train_controls = {"ceu":f["train_controls_ceu"][()],"yri":f["train_controls_yri"][()],
                       "meta":np.append(f["train_controls_ceu"][()],f["train_controls_yri"][()]),
                       "la":f["train_controls_yri"][()]}
        labels_all = f["labels"][()]
        f.close()
    else: # else load subset of YRI cases, remove in future (training accuracy will be underestimated)
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
    # load testing data
    all_testing = h5py.File(f"{prefix}true_prs/prs_m_{m}_h2_{h2}.hdf5","r")['test_data'][()]
    # initialize testing by ancestry dictionary
    testing =  {"ceu":np.array([]),"yri":np.array([]),"admix":np.array([])}
    # loop through testing
    for ind in all_testing:
    	# decode labels
        label = labels_all[ind].decode("utf-8")
        # if ceu in labels then individual is European
        if "ceu" in label: testing["ceu"] = np.append(testing["ceu"],[ind]).astype(int)
        # if yri in labels then individual is African
        elif "yri" in label: testing["yri"]= np.append(testing["yri"],[ind]).astype(int)
        # else individual is admixed
        else: testing["admix"]= np.append(testing["admix"],ind).astype(int)

    f.close()
    # open true prs file
    true_prs = h5py.File(f"{prefix}true_prs/prs_m_{m}_h2_{h2}.hdf5","r")['X'][()]
    # open empirical prs file
    emp_prs = h5py.File(f"{prefix}emp_prs/emp_prs_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{snp_selection}_"+\
    					f"snps_{len(train_cases[snp_selection])}cases"+\
                        f"_{snp_weight}_weights_{len(train_cases[snp_weight])}cases.hdf5","r")['X'][()]
    # return all loaded data
    return true_prs,emp_prs,anc,testing,train_cases,train_controls,labels_all
