import matplotlib.pyplot as plt
import seaborn as sns
import sys
import gzip
import multiprocessing as mp

sys.path.insert(0,"/Users/taylorcavazos/repos/Local_Ancestry_PRS/code/")
sys.path.insert(0,"/Users/taylorcavazos/Documents/Prelim_Quals/Aim1")
from output_emp_prs import *

m = 1000
h2 = 0.67
path_tree = "/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/trees/tree_sub_CEU_1.95e5.hdf"
path_tree_all = "/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/trees/tree_YRI_5e4_CEU_2e5_chr22.hdf5"
sample_map_all = pd.read_csv("/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/trees/CEU_YRI_sample_map.txt",sep="\t",header=None)
eur_only = sample_map_all[sample_map_all.iloc[:,1]=="CEU"].index
tree = msprime.load(path_tree)
tree_all = msprime.load(path_tree_all)
tree_all_filt = msprime.load(path_tree_all)
#tree_eur_all = msprime.load(path_tree_all).simplify(samples=eur_only,filter_sites=False)
#tree_eur_all_filt = msprime.load(path_tree_all).simplify(samples=eur_only,filter_sites=True)


n_sites = tree.num_sites
bonf_p = 0.05/n_sites

sum_stats = pd.read_csv("/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/comm_maf_0.01_sum_stats_m_{}_h2_{}_fdr.txt".format(m,h2),sep="\t",index_col=0)

calc_emp_prs(tree,tree_all,tree_all_filt,5e-8,sum_stats,vcf_file="/Users/taylorcavazos/repos/other_tools/admixture-data/output/admix_afr_amer.query.vcf")

