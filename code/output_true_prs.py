import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
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
import argparse

def return_diploid_genos(variant,tree):
    genos_diploid = np.sum(variant.reshape([1,int(tree.num_samples/2),2]),axis=-1)
    return genos_diploid

def simulate_genos(rmap, chrom, n_samps):
    tree = msprime.simulate(sample_size=(2*n_samps), Ne=1e4, mutation_rate=2e-8,
                        random_seed=53195, recombination_map = rmap,)
    return tree

def compute_effects(m, h2,num_sites):
    causal_inds = np.linspace(0, num_sites, m, dtype=int,endpoint=False)
    effect_sizes = np.random.normal(loc=0, scale=(h2/m),size=m)
    
    all_effects = np.zeros(num_sites)
    np.put(all_effects,causal_inds,effect_sizes)
    return all_effects

def compute_true_PRS(genos_diploid, all_effects,h2):
    X = np.dot(genos_diploid,all_effects)
    return X

def var_iterate_compute_prs(all_effects, h2, tree):
    non0 = np.where(all_effects!=0)[0]
    X_sum = 0
    i = 0
    for variant in tree.variants():
        if variant.site.id in non0:
            var_dip = return_diploid_genos(variant.genotypes,tree)
            X_sum+=compute_true_PRS(var_dip.T, all_effects[i],h2)
        i+=1
    return X_sum

def calc_prs_vcf(vcf_file,effects,n_admix):
    prs = np.zeros(n_admix)
    with open(vcf_file) as f:
        ind=0
        for line in f:
            if line[0] != "#":
                if effects[ind] != 0:
                    data = line.split("\t")[9:]
                    genotype = np.array([np.array(hap.split("|")).astype(int).sum() for hap in data])
                    prs=prs+genotype*effects[ind]
                ind+=1
    return prs

def simulate_true_prs(it,m,h2,n_sites,path_tree,n_admix,admix_vcf=None):
    all_effects = compute_effects(m, h2,n_sites)
    tree = msprime.load(path_tree)
    prs = var_iterate_compute_prs(all_effects,h2,tree)
    if admix_vcf == None:
        return prs
    else:
        prs_admix= calc_prs_vcf(admix_vcf,all_effects,n_admix) # Do something
        return np.append(prs,prs_admix)

def summarize_true_prs(true_prs_sim,n_samps,h2):
    X = np.average(true_prs_sim,axis=0).reshape(n_samps)
    Zx = (X-np.mean(X))/np.std(X)
    G = np.sqrt(h2)*Zx
    return X,Zx,G

def main(path_tree,m,h2,outfile_prefix,iters,n_admix,admix_vcf=None):
    #rmap = msprime.RecombinationMap.read_hapmap("hapmap/genetic_map_GRCh37_chr22.txt")
    chrom = "22"
    tree = msprime.load(path_tree)
    n_samps = int(tree.num_samples/2)
    n_sites = tree.num_sites
    pool = Pool(4)
    print("Calculating true PRS by averaging over {} times".format(iters))
    sub_func=partial(simulate_true_prs, m=m, h2=h2,n_sites=n_sites,n_admix=n_admix,
        path_tree=path_tree,admix_vcf=admix_vcf)
    true_prs_sim = pool.map(sub_func,range(iters))
    pool.close()
    pool.join()
    print("Saving data to file",flush=True)
    if admix_vcf == None:
        X,Zx,G = summarize_true_prs(true_prs_sim,n_samps,h2)

        with h5py.File(outfile_prefix+'.hdf5', 'w') as f:
            f.create_dataset("X",(n_samps,),dtype=float,data=X)
            f.create_dataset("Zx",(n_samps,),dtype=float,data=Zx)
            f.create_dataset("G",(n_samps,),dtype=float,data=G)

    else:
        sample_ids_eur = ["msp_{}".format(i) for i in range(0,n_samps)]
        sample_ids_admix = list(pd.read_csv("admixed_data/output/admix_afr_amer.prop.anc",sep="\t",index_col=0).index)
        sample_ids = np.append(sample_ids_eur,sample_ids_admix)
        n_samps_ad = len(sample_ids_admix)
        n_all = len(sample_ids)
        # X,Zx,G = summarize_true_prs(true_prs_sim[0],n_samps,h2)

        # with h5py.File(outfile_prefix+'.hdf5', 'w') as f:
        #     f.create_dataset("X",(n_samps,),dtype=float,data=X)

        X_ad,Zx_ad,G_ad = summarize_true_prs(true_prs_sim,n_all,h2)
        with h5py.File(outfile_prefix+'.hdf5', 'w') as f:
            f.create_dataset("labels",(n_all,),data=sample_ids.astype("S"))
            f.create_dataset("X",(n_all,),dtype=float,data=X_ad)
            f.create_dataset("Zx",(n_all,),dtype=float,data=Zx_ad)
            f.create_dataset("G",(n_all,),dtype=float,data=G_ad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation of population trees")
    parser.add_argument("--tree",help="path to tree file for creating true PRS", type=str, 
        default="trees/tree_CEU_GWAS_nofilt.hdf")
    parser.add_argument("--numADMIX", help="number of ADMIX individuals", type=int, default=5000)
    parser.add_argument("--m",help="number of causal variants", type=int, default=1000)
    parser.add_argument("--h2",help="heritability", type=float, default=0.67)
    parser.add_argument("--iters", help="iteration number", type=int, default=100)
    parser.add_argument("--admixVCF", help="VCF file for admixed population",
        type=str,default="admixed_data/output/admix_afr_amer.query.vcf")

    args = parser.parse_args()
    main(args.tree, args.m, args.h2, "true_prs/m_{}_h2_{}".format(args.m, args.h2), args.iters,args.numADMIX, 
        admix_vcf = args.admixVCF)