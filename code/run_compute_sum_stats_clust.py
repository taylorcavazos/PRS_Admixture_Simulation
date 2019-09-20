import numpy as np, pandas as pd
import msprime, tqdm
import sys, argparse
import gzip, h5py, os
import multiprocessing as mp
import scipy.stats as stats, math
import statsmodels.api as sm

def return_diploid_genos(variant,tree):
    genos_diploid = np.sum(variant.reshape([1,int(tree.num_samples/2),2]),axis=-1)
    return genos_diploid

def gwas(case_ref,case_alt,control_ref,control_alt):
    cont_tabl = np.array([[case_alt,case_ref],[control_alt,control_ref]]).reshape(2,2)
    # OR,pval = stats.fisher_exact(cont_tabl)
    result = sm.stats.Table2x2(cont_tabl)
    return result.oddsratio, result.oddratio_pvalue()

def compute_maf(path_tree):
    tree = msprime.load(path_tree)
    n_sites = tree.num_sites
    maf = []
    pbar = tqdm.tqdm(total=tree.num_sites)
    for variant in tree.variants():
        genos_diploid = return_diploid_genos(variant.genotypes,tree)
        freq = np.sum(genos_diploid,axis=1)/(2*genos_diploid.shape[1])
        if freq < 0.5: maf.append(freq)
        else: maf.append(1-freq)
        pbar.update(1)
    np.savetxt("emp_prs/maf.txt",maf)
    return maf

def output_sum_stats(variant,path_tree, train_cases, train_controls):
    tree = msprime.load(path_tree)

    genos_diploid_common = return_diploid_genos(variant.genotypes,tree)
    genos_diploid_common_cases = genos_diploid_common[:,train_cases]
    genos_diploid_common_controls = genos_diploid_common[:,train_controls]

    case_alt = np.sum(genos_diploid_common_cases>0,axis=1)
    case_ref = np.sum(genos_diploid_common_cases==0,axis=1)

    control_alt = np.sum(genos_diploid_common_controls>0,axis=1)
    control_ref = np.sum(genos_diploid_common_controls==0,axis=1)
    OR,pval = gwas(case_ref,case_alt,control_ref,control_alt)
    return [variant.site.id,OR,pval]

def main(path_tree,m,h2):

    f = h5py.File('true_prs/train_test_m_{}_h2_{}.hdf5'.format(m,h2), 'r')
    train_cases,train_controls = f["train_cases"][()].astype(int), f["train_controls"][()].astype(int)
    f.close()

    if os.path.isfile("emp_prs/maf.txt"): maf = np.loadtxt("emp_prs/maf.txt")
    else: maf = compute_maf(path_tree)
    common_vars = np.where(np.array(maf) >= 0.01)[0]

    tree = msprime.load(path_tree)
    
    pool = mp.Pool(10)
    results = [pool.apply_async(output_sum_stats, args=(variant,path_tree,train_cases,train_controls)) for variant in tree.variants() if variant.site.id in common_vars]
    output = [p.get() for p in results]
    sum_stats = pd.DataFrame(output,columns=["var_id","OR","p-value"]).set_index("var_id")
    sum_stats = sum_stats.replace([np.inf, -np.inf], np.nan)
    sum_stats.dropna(inplace=True)
    sum_stats = sum_stats.sort_index()
    sum_stats.to_csv("emp_prs/comm_maf_0.01_sum_stats_m_{}_h2_{}.txt".format(m,h2),sep="\t",index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create summary statistics")
    parser.add_argument("--tree",help="path to tree file for creating true PRS", type=str, 
        default="trees/tree_CEU_GWAS_nofilt.hdf")
    parser.add_argument("--m",help="number of causal variants", type=int, default=1000)
    parser.add_argument("--h2",help="heritability", type=float, default=0.67)
    
    args = parser.parse_args()
    main(args.tree, args.m, args.h2)

