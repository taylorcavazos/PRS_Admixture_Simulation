import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sys
import gzip
import multiprocessing as mp
import glob
import numpy as np
import h5py
from scipy import stats
import msprime
import pandas as pd
from scipy.stats import chi2

sim = sys.argv[1]
m = sys.argv[2]
h2 = sys.argv[3]


sns.set_context("notebook")
sns.set_style("ticks")

if "REGRESS" not in sim:
    sum_stats = pd.read_csv("emp_prs/comm_maf_0.01_sum_stats_m_{}_h2_{}.txt".format(m,h2),index_col=0,sep="\t")

else: 
    sum_stats = pd.read_csv("emp_prs/comm_maf_0.01_sum_stats_m_{}_h2_{}_REGRESS.txt".format(m,h2),index_col=0,sep="\t")

chisq = chi2.ppf(1-sum_stats["p-value"],1)
lam_gc = np.median(chisq)/chi2.ppf(0.5,1)
pvals = sum_stats["p-value"].values
expected_p = (stats.rankdata(pvals,method="ordinal")+0.5)/(len(pvals)+1)

plt.figure(figsize=(5,5))
plt.scatter(-1*np.log10(expected_p), -1*np.log10(pvals),s=20)
plt.plot(sorted(-1*np.log10(expected_p)),sorted(-1*np.log10(expected_p)),c="black",linestyle="--")
plt.text(2.5,200,"$\lambda$ = {}".format(np.round(lam_gc,2)),fontsize=20)
plt.xlabel("Expected -log10 P-value",fontsize=16)
plt.ylabel("Observed -log10 P-Value",fontsize=16)
sns.despine()
plt.savefig("plots/qq_plot_m_{}_h2_{}_{}.png".format(m,h2,sim),type="png",bbox_inches="tight",dpi=400)
