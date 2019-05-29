import matplotlib.pyplot as plt
import seaborn as sns
import sys
import gzip
import multiprocessing as mp

sys.path.insert(0,"/Users/taylorcavazos/repos/Local_Ancestry_PRS/code/")
sys.path.insert(0,"/Users/taylorcavazos/Documents/Prelim_Quals/Aim1")

from sim_out_of_africa import *
from output_true_prs import *
from compute_sum_stats import *

m = 1000
h2 = 0.67
path_tree = "/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/trees/tree_sub_CEU_1.95e5.hdf"

tree = msprime.load(path_tree)
n_sites = tree.num_sites

f = h5py.File('/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/train_test_CEU_m_{}_h2_{}.hdf5'.format(m,h2), 'r')
train_cases,train_controls = f["train_cases"][()].astype(int), f["train_controls"][()].astype(int)
test_cases,test_controls = f["test_cases"][()].astype(int), f["test_controls"][()].astype(int)
f.close()

maf = np.loadtxt("/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/maf.txt")
common_vars = np.where(np.array(maf) >= 0.01)[0]

def output_sum_stats(variant,path_tree):
    tree = msprime.load(path_tree)
    print(variant.site.id,flush=True)

    genos_diploid_common = return_diploid_genos(variant.genotypes,tree)
    genos_diploid_common_cases = genos_diploid_common[:,train_cases]
    genos_diploid_common_controls = genos_diploid_common[:,train_controls]

    case_alt = np.sum(genos_diploid_common_cases>0,axis=1)
    case_ref = np.sum(genos_diploid_common_cases==0,axis=1)

    control_alt = np.sum(genos_diploid_common_controls>0,axis=1)
    control_ref = np.sum(genos_diploid_common_controls==0,axis=1)
    OR,pval = gwas(case_ref,case_alt,control_ref,control_alt)
    return [variant.site.id,OR,pval]

pool = mp.Pool(14)
results = [pool.apply_async(output_sum_stats, args=(variant,path_tree)) for variant in tree.variants() if variant.site.id in common_vars]
output = [p.get() for p in results]
sum_stats = pd.DataFrame(output,columns=["var_id","OR","p-value"]).set_index("var_id")
sum_stats = sum_stats.replace([np.inf, -np.inf], np.nan)
sum_stats.dropna(inplace=True)
sum_stats = sum_stats.sort_index()
sum_stats.to_csv("/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/comm_maf_0.01_sum_stats_m_{}_h2_{}.txt".format(m,h2),sep="\t",index=True)
