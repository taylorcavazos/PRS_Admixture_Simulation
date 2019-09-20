import numpy as np, pandas as pd
import tqdm,time,msprime,h5py
from multiprocessing import Pool, Process, Manager
from functools import partial
import threading, copy, pickle
import scipy.stats as stats, math
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from itertools import islice, count
import sys, argparse, os

def return_diploid_genos(tree,variant):
    genos_diploid = np.sum(variant.reshape([1,int(tree.num_samples/2),2]),axis=-1)
    return genos_diploid

def find_ld_sites(tree_sequence,
                  focal_vars,
                  var2mut_dict,mut2var_dict,
                  max_distance=1e6,
                  r2_threshold=0.5,
                  num_threads=8):
    results = {}
    num_threads = min(num_threads, len(focal_vars))

    def thread_worker(thread_index):
        ld_calc = msprime.LdCalculator(tree_sequence)
        chunk_size = int(math.ceil(len(focal_vars) / num_threads))
        start = thread_index * chunk_size
        for focal_var in focal_vars[start: start + chunk_size]:
            focal_mutation = var2mut_dict.get(focal_var)
            np.seterr(under='ignore')
            a = ld_calc.get_r2_array(
                focal_mutation, max_distance=max_distance,
                direction=msprime.REVERSE)
            a[np.isnan(a)] = 0
            rev_indexes = focal_mutation - np.nonzero(a >= r2_threshold)[0] - 1
            a = ld_calc.get_r2_array(
                focal_mutation, max_distance=max_distance,
                direction=msprime.FORWARD)
            a[np.isnan(a)] = 0
            fwd_indexes = focal_mutation + np.nonzero(a >= r2_threshold)[0] + 1
            indexes = np.concatenate((rev_indexes[::-1], fwd_indexes))
            indexes = [mut2var_dict.get(ind) for ind in indexes if mut2var_dict.get(ind)!=None]
            results[mut2var_dict.get(focal_mutation)] = indexes


    threads = [
    threading.Thread(target=thread_worker, args=(j,)) for j in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results

def ld_clump(sorted_vars, ld_struct):
    clumped_prs = [sorted_vars[0]]

    for v in range(1, len(sorted_vars)):
        add = True
        i = 0
        while add and i < len(clumped_prs):
            if sorted_vars[v] in ld_struct.get(clumped_prs[i]):
                add = False
            i+=1
        if add == True:
            clumped_prs.append(sorted_vars[v])
    return clumped_prs

def var_iterate_compute_emp_prs(tree,clumped_prs_vars,sum_stats):
    X_sum = 0

    print("Calculating emprirical PRS tree")
    pbar = tqdm.tqdm(total=tree.num_sites)
    for variant in tree.variants():
        if variant.site.id in clumped_prs_vars:

            var_dip = return_diploid_genos(tree,variant.genotypes)
            log_OR = np.log(sum_stats.loc[variant.site.id,"OR"])
            if np.isnan(log_OR)!=True and np.isinf(log_OR)!=True:
                X_sum+=np.dot(var_dip.T, log_OR)

        pbar.update(1)
    return X_sum

def calc_prs_vcf(vcf_file,sum_stats,clumped_prs_vars,n_admix=4000):
    print("Calculating empirical PRS vcf")
    prs = np.zeros(n_admix)
    with open(vcf_file) as f:
        ind=0
        for line in f:
            if line[0] != "#":
                if ind in clumped_prs_vars:
                    data = line.split("\t")[9:]
                    genotype = np.array([np.array(hap.split("|")).astype(int).sum() for hap in data])
                    prs=prs+genotype*np.log(sum_stats.loc[ind,"OR"])
                ind+=1
    return prs

def calc_prs_vcf_anc(vcf_file,sum_stats,clumped_prs_vars,sample_ids,n_admix=4000):
    # anc = pd.read_csv("/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/CEU_counts_prs_vars.txt",sep="\t",index_col=0)
    print("Calculating empirical PRS vcf")
    prs = np.zeros(n_admix)
    with open(vcf_file) as f:
        ind=0
        for line in f:
            if line[0] != "#":
                if ind in clumped_prs_vars:
                    data = line.split("\t")[9:]
                    genotype = np.array([np.array(hap.split("|")).astype(int).sum() for hap in data])
                    var_weighted=[]
                    for g in range(0,len(genotype)):
                        # if anc.loc[ind,sample_ids[g]] >= 1:
                        var_weighted.append(genotype[g]*(np.log(sum_stats.loc[ind,"OR"])))
                        # else:
                        #     var_weighted.append(genotype[g]*0)
                    prs=prs+np.array(var_weighted)
                ind+=1
    return prs

def main(tree_path, tree_ld_path, p_thresh, sum_stats,vcf_file=None,m=1000,h2=0.67,r2=0.2,name=None):
    tree, tree_LD = msprime.load(tree_path), msprime.load(tree_ld_path)

    var2mut, mut2var = {}, {}
    for mut in tree_LD.mutations():
        mut2var[mut.id]=mut.site
        var2mut[mut.site]=mut.id

    tree_LD_filt = tree_LD.simplify(filter_sites=True)
    prs_vars = sum_stats[sum_stats["p-value"] < p_thresh].sort_values(by=["p-value"]).index
    prs_vars_ld_pres = [var for var in prs_vars if var in var2mut.keys()]
    ld_struct = find_ld_sites(tree_LD_filt,prs_vars_ld_pres,var2mut,mut2var,r2_threshold=r2)

    clumped_prs_vars = ld_clump(prs_vars_ld_pres,ld_struct)
    if name != None:
        np.savetxt("emp_prs/clumped_prs_vars_m_{}_h2_{}_ld_{}_r2_{}_p{}.txt".format(m,h2,name,r2,p_thresh),clumped_prs_vars)
    else:
        np.savetxt("emp_prs/clumped_prs_vars_m_{}_h2_{}_ld_{}_r2_{}_p{}.txt".format(m,h2,tree_ld_path[11:14],r2,p_thresh),clumped_prs_vars)
    print("Number of indendent signals is {}".format(len(clumped_prs_vars)),flush=True)
    print("Computing empirical PRS",flush=True)
    time.sleep(1)

    if type(vcf_file) != type(None):
        sample_ids_admix = pd.read_csv("admixed_data/output/admix_afr_amer.prop.anc",sep="\t",index_col=0).index
        X_emp_admix = calc_prs_vcf(vcf_file,sum_stats,clumped_prs_vars,n_admix=len(sample_ids_admix))
        
        n_samps = int(tree.num_samples/2)
        sample_ids_eur = ["msp_{}".format(i) for i in range(0,n_samps)]
        X_emp = var_iterate_compute_emp_prs(tree,clumped_prs_vars,sum_stats)

        sample_ids = np.append(sample_ids_eur,sample_ids_admix)
        n_samps_ad = len(sample_ids_admix)
        n_all = len(sample_ids)
        X_emp_all = np.append(X_emp,X_emp_admix)

        print("Writing output file")
        if name != None:
            with h5py.File('emp_prs/emp_prs_m_{}_h2_{}_ld_{}_r2_{}_p{}.hdf5'.format(m,h2,name,r2,p_thresh), 'w') as f:
                f.create_dataset("labels",(n_all,),data=np.array(sample_ids).astype("S"))
                f.create_dataset("X",(n_all,),dtype=float,data=X_emp_all)
        else:
            with h5py.File('emp_prs/emp_prs_m_{}_h2_{}_ld_{}_r2_{}_p{}.hdf5'.format(m,h2,tree_ld_path[11:14],r2,p_thresh), 'w') as f:
                f.create_dataset("labels",(n_all,),data=np.array(sample_ids).astype("S"))
                f.create_dataset("X",(n_all,),dtype=float,data=X_emp_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculates empirical prs")
    parser.add_argument("--tree",help="path to tree file for creating true PRS", type=str, 
        default="trees/tree_CEU_GWAS_nofilt.hdf")
    parser.add_argument("--treeLD", help="path to tree used for LD", type=str, default="trees/tree_all.hdf")
    parser.add_argument("--m",help="number of causal variants", type=int, default=1000)
    parser.add_argument("--h2",help="heritability", type=float, default=0.67)
    parser.add_argument("--pval", help="p-value for cutoff", type=float, default=0.00000005)
    parser.add_argument("--r2", help="p-value for cutoff", type=float, default=0.2)
    parser.add_argument("--admixVCF", help="VCF file for admixed population",
        type=str,default="admixed_data/output/admix_afr_amer.query.vcf")
    parser.add_argument("--name",help="LD name", type=str,default=None)
    args = parser.parse_args()
    sum_stats =pd.read_csv("emp_prs/comm_maf_0.01_sum_stats_m_{}_h2_{}.txt".format(args.m,args.h2),sep="\t",index_col=0)
    sum_stats.loc[sum_stats.OR==0,"OR"] = 1
    main(args.tree,args.treeLD,args.pval,sum_stats,vcf_file=args.admixVCF,m=args.m, h2=args.h2, r2=args.r2,name=args.name)
