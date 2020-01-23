import pandas as pd
import seaborn as sns
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools

def plot_maf_bins(filepat="../results/summary/maf_sims/maf_bins_m_1000_h2_0.5_*"):
	all_df = pd.DataFrame(columns=["Sim","Pop","G1","G2","G3","G4","G5","G6"])
	for file in glob.glob(filepat):
	    df = pd.read_csv(file,sep="\t")
	    all_df = all_df.append(df,ignore_index=True)

	df_long = pd.melt(all_df.iloc[:,1:],"Pop",var_name="G")

	plt.figure(figsize=(10,5))
	plt.rc('ytick',labelsize=18)
	ax = sns.barplot(x="G",hue="Pop",y="value",data=df_long,palette=["#103c42","#05a19c"],hue_order=["European","Admixed"])

	plt.xlabel("MAF",fontsize=20)
	plt.ylabel("Proportion of PRS Variants",fontsize=20)

	ax.set_xticklabels(labels=["0 - 0.01","0.01 - 0.1","0.1 - 0.2","0.2 - 0.3","0.3 - 0.4","0.4 - 0.5"],fontsize=18)

	sns.despine()
	h, l = ax.get_legend_handles_labels()
	plt.legend(h,["European", "Admixed"],frameon=False,fontsize=18)
	return