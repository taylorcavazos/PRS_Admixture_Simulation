import pandas as pd
import seaborn as sns
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools

cols = ["test_EUR_corr","ADMIX_high_eur_corr","ADMIX_mid_eur_corr","ADMIX_low_eur_corr","weight"]
my_rgbs = np.array(["#103c42","#02576c","#05a19c","#ffe837"])

sns.set_context("notebook")
sns.set(style = 'whitegrid', font_scale = 2.5)

def load_all_weight_summary_data():
    full_df_prs = pd.DataFrame(columns = ["weight","train_corr","test_EUR_corr","test_ADMIX_corr",
                                 "ADMIX_low_eur_corr","ADMIX_mid_eur_corr","ADMIX_high_eur_corr"])

    for f in glob.glob("../results/summary/corrs/*/*"):
        df = pd.read_csv(f,sep="\t",index_col=0)
        df.index = [f.split("_")[-1].split(".")[0]]
        df.loc[:,"sim"] = f.split("_")[-1].split(".")[0]
        
        df.loc[:,"m"] = int(f.split("_")[4])
        df.loc[:,"h2"] = float(f.split("_")[6])
        df.loc[:,"pval"] = float(f.split("_")[10])
        df.loc[:,"r2"] = float(f.split("_")[8])
        
        if "LA_weights" in f:
            df.loc[:,"weight"] = "Local ancestry \nspecific"
        elif "yri_weights" in f:
            df.loc[:,"weight"] = "African"
        else: df.loc[:,"weight"] = "European"
            
        full_df_prs = full_df_prs.append(df,ignore_index=False,sort=True)
    
    return full_df_prs


def plot_correlation(data,axis,palette=np.array(["#103c42","#02576c","#05a19c","#ffe837"]),title=""):
    g = sns.violinplot(data = data,palette=palette,ax=axis,cut=0)
    axis.set_title(title)
    return g

def plot_correlation_single_eur_weights(data,ax,m=500,h2=0.5):
    data = data[data["weight"] == "European"]
    g = plot_correlation(data.loc[(data["m"]==m)&(data["h2"]==h2),cols],ax)
    ax.set_title(r"$m = {}, h^2 = {}$".format(m,h2))
    ax.set_ylabel("Pearson's correlation")

    names = ["European-only","CEU $>$ 80%","80% $\geq$ CEU $>$ 20%", "CEU $\leq$ 20%"]
    g.set_xticklabels(names,horizontalalignment="right",rotation=40)

def plot_correlation_all_params_eur_weights():
    m = [200,500,1000]
    h2 = [0.33,0.5,0.67]
    pairs = list(itertools.product(h2,m))

    data = load_all_weight_summary_data()
    data = data[data["weight"] == "European"]

    fig,ax = plt.subplots(3,3,figsize=(20,20),sharey = True, sharex = True)
    for num, axes in enumerate(ax.T.ravel()):
        g = plot_correlation(data.loc[(data["m"]==pairs[num][1])&(data["h2"]==pairs[num][0]),cols],
                        axes)
        g.set_xticklabels("")
    
    for i in range(3):
        ax[i,0].set_ylabel(r"$m = {}$".format(m[i]),fontsize=36,rotation=0,labelpad=180)
        ax[0,i].set_title(r"$h^2 = {}$".format(h2[i]),fontsize=36)

    for axes in ax.flatten():
        axes.tick_params(axis='both', which='major', labelsize=36)
        axes.tick_params(axis='both', which='minor', labelsize=36)

    plt.ylim(0.2,0.9)

    legend_dict = dict(zip(["European-only","CEU $>$ 80%","80% $\geq$ CEU $>$ 20%", "CEU $\leq$ 20%"],my_rgbs))
    patchList = []
    for key in legend_dict:
            data_key = mpatches.Patch(color=legend_dict[key], label=key)
            patchList.append(data_key)

    fig.legend(handles=patchList,bbox_to_anchor=(0.6,0.01),loc=8,ncol=4,fancybox=False,frameon=False,fontsize=30,
              columnspacing=1,handletextpad=0.4)
    sns.despine(left=True)
    fig.text(x=0.05,y=0.59,s=r"Pearson's correlation",rotation=90,fontsize=36)
    fig.suptitle("Correlation Between True and Empirical PRS Across 50 Simulations",y=0.95,x=0.5,fontsize=36)
    return fig

def plot_correlation_single_all_weights(data,ax,m=500,h2=0.5):
    long_df = data.melt(value_vars=cols[:-1],id_vars=["weight","sim","m","h2"])
    g = sns.boxplot(y="variable",x="value",hue="weight",hue_order = ["European","African","Local ancestry \nspecific"],
                   data=long_df[(long_df["m"]==m)&(long_df["h2"]==h2)],ax=ax,palette=["#B4B9BF","#fe6845","#91bd3a"])
    g.set_xticks(np.arange(0,1.2,0.2))
    g.set_yticklabels(["European-only","CEU$>$80%","80%$\geq$CEU$>$20%", "CEU$\leq$20%"],
                     rotation=0)
    legend = plt.legend(frameon=False,title="PRS Weights",fancybox=False,loc=(1,0.3))
    plt.ylabel("")
    plt.xlabel("Pearson's correlation")
    ax.set_title(r"$m = {}, h^2 = {}$".format(m,h2))
    
    return

def plot_correlation_all_params_all_weights():
    m = [200,500,1000]
    h2 = [0.33,0.5,0.67]
    pairs = list(itertools.product(h2,m))

    data = load_all_weight_summary_data()
    long_df = data.melt(value_vars=cols[:-1],id_vars=["weight","sim","m","h2"])

    fig,ax = plt.subplots(3,3,figsize=(30,20),sharey = True, sharex = True)

    for num, axes in enumerate(ax.T.ravel()):
        g = sns.boxplot(y="variable",x="value",hue="weight",hue_order = ["European","African","Local ancestry \nspecific"],
                       data=long_df[(long_df["m"]==pairs[num][1])&(long_df["h2"]==pairs[num][0])],ax=axes,palette=["#B4B9BF","#fe6845","#91bd3a"])
        axes.get_legend().set_visible(False)
        axes.set_ylabel("")
        axes.set_xlabel("")
        axes.set_yticklabels("")
        axes.tick_params(axis="both",labelsize=36)

    for i in range(3):
        ax[i,0].set_ylabel(r"$m = {}$".format(m[i]),fontsize=36,rotation=0,labelpad=90)
        ax[0,i].set_title(r"$h^2 = {}$".format(h2[i]),fontsize=36)

    legend_dict = dict(zip(["European","African","Local ancestry\nspecific"],["#B4B9BF","#fe6845","#91bd3a"]))
    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

    fig.text(x=0.5,y=-0.03,s=r"Pearson's correlation",rotation=0,fontsize=36)
    legend = fig.legend(bbox_to_anchor=(1.19,0.7),handles=patchList,fancybox=False,frameon=False,fontsize=36,title="PRS Weights")
    plt.setp(legend.get_title(),fontsize=36)
    plt.tight_layout()

    return fig

def load_data_decreasing_yri(m,h2,pop):
    sample_sizes = [100,500,1000,5000,7000,9000,10000]
    data = load_all_weight_summary_data()
    # Loading data for full sample size
    # afr_weight = data.loc[(data["m"]==m)&(data["h2"]==h2)&(data["weight"]=="African"),pop]
    eur_weight = data.loc[(data["m"]==m)&(data["h2"]==h2)&(data["weight"]=="European"),pop]
    # LA_weight = data.loc[(data["m"]==m)&(data["h2"]==h2)&(data["weight"]=="Local ancestry \nspecific"),pop]

    df = pd.DataFrame(columns=["simulation","sample_size","weighting","correlation"])

    for size in sample_sizes:
        df = df.append(pd.DataFrame({"simulation":list(eur_weight.index),
                            "sample_size":size,"weighting":"CEU",
                            "correlation":list(eur_weight.values),}),ignore_index=True,sort=False)

    num_dict = {"simulation":[],"sample_size":[],"weighting":[],"correlation":[]}
    for f in glob.glob("../results/summary/yri_train_decrease/*"):
        sim = f.split("_")[-1].split(".")[0]
        num = int(f.split("_")[-2])
        weight = f.split("_")[-5]
        sub_df = pd.read_csv(f,sep="\t",index_col=0)
        
        num_dict["simulation"].append(sim)
        num_dict["sample_size"].append(num)
        num_dict["weighting"].append(weight.upper())
        num_dict["correlation"].append(sub_df.loc["vals",pop])

    df = df.append(pd.DataFrame(num_dict),ignore_index=True,sort=False)
    # df = df.append(pd.DataFrame({"simulation":list(afr_weight.index),
    #                     "sample_size":10000,"weighting":"YRI",
    #                     "correlation":list(afr_weight.values),}),ignore_index=True,sort=False)
    # df = df.append(pd.DataFrame({"simulation":list(LA_weight.index),
    #                     "sample_size":10000,"weighting":"LA",
    #                     "correlation":list(LA_weight.values),}),ignore_index=True,sort=False)

    df = df.sort_values(by="sample_size")
    df["sample_size"] = df["sample_size"].astype(str)
    return df

def plot_correlation_decreasing_yri(m=1000,h2=0.5,pop="ADMIX_low_eur_corr", ax=None,
                                    title="PRS weighting by sample size for CEU $\leq$ 20%"):
    sns.set_style("ticks")
    df = load_data_decreasing_yri(m,h2,pop)
    fig,ax = plt.subplots(figsize=(12,8))
    g = sns.lineplot(x="sample_size",y="correlation",hue="weighting",data=df,ax=ax,sort=False
                 ,palette=["#B4B9BF","#fe6845","#91bd3a"],hue_order=["CEU","YRI","LA"],legend=False,linewidth=2.5)

    ax.set_ylim(0.2,0.8)

    ax.set_ylabel("Pearson's correlation",fontsize=26)
    ax.set_xlabel("# of YRI Cases",fontsize=26)
    ax.set_title(title,fontsize=26)

    plt.legend(loc=(1,0.8),frameon=False,labels=["CEU","YRI","LA"],fontsize=20)

    sns.despine()
    return

def plot_correlation_decreasing_yri_allPops(m=1000,h2=0.5):
    sns.set(style = 'ticks', font_scale = 3)
    pops = ["ADMIX_low_eur_corr","ADMIX_mid_eur_corr","ADMIX_high_eur_corr"]
    title = ["CEU $\leq$ 20%","20%$<$CEU$\leq$80%","CEU$>$80%"]
    fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(30,12),sharey=True)
    for ind, ax in enumerate(axes):
        df = load_data_decreasing_yri(m,h2,pops[ind])
        g = sns.lineplot(x="sample_size",y="correlation",hue="weighting",data=df,ax=ax,sort=False
                 ,palette=["#B4B9BF","#fe6845","#91bd3a"],hue_order=["CEU","YRI","LA"],legend=False,linewidth=2.5)

        ax.set_ylim(0.2,0.8)
        ax.set_xlabel("# of YRI Cases")
        ax.set_ylabel("")

    # ax.set_ylabel("Pearson's correlation",fontsize=26)
    # ax.set_xlabel("# of YRI Cases",fontsize=26)
        ax.set_title(title[ind])

    axes[1].legend(loc=(0,-0.4),frameon=True,labels=["CEU","YRI","LA"],ncol=3,title="PRS Weights",fancybox=True)
    axes[0].set_ylabel("Pearson's correlation")
    sns.despine()
    fig.tight_layout(pad=0)
    return