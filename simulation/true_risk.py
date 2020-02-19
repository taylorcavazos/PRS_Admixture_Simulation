import numpy as np, pandas as pd
import msprime,h5py

def simulate_true_prs(m,h2,n_admix,prefix="../output/sim1/",
                        path_tree_CEU="trees/tree_CEU_GWAS_nofilt.hdf",
                        path_tree_YRI="trees/tree_YRI_GWAS_nofilt.hdf",
                        vcf_file="admixed_data/output/admix_afr_amer.query.vcf"):
    """
    TODO.

    Parameters
    ----------
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    n_admix : msprime.simulations.RecombinationMap
        Recombination map for a reference chromosome
    prefix : str, optional
        Output file path
    path_tree_CEU : str, optional
        Path to simulated tree containing CEU individuals
    path_tree_YRI : str, optional
        Path to simulated tree containing YRI individuals
    vcf_file : str, optional
        VCF file path with admixed genotypes

    """
    tree_CEU = msprime.load(path_tree_CEU)
    tree_YRI = msprime.load(path_tree_YRI)
    all_effects = _compute_effects(m, h2,tree_CEU.num_sites)
    
    prs_CEU = _calc_prs_tree(all_effects,h2,tree_CEU)
    prs_YRI = _calc_prs_tree(all_effects,h2,tree_YRI)
    prs_admix= _calc_prs_vcf(prefix+vcf_file,all_effects,n_admix)
    prs_comb = np.concatenate((prs_CEU,prs_YRI,prs_admix),axis=None)

    _write_true_prs(tree_ceu,tree_yri,m,h2,prs_comb,all_effects,prefix)

    return

def _return_diploid_genos(variant,tree):
    genos_diploid = np.sum(variant.reshape([1,int(tree.num_samples/2),2]),axis=-1)
    return genos_diploid

def _compute_effects(m, h2,num_sites):
    causal_inds = np.linspace(0, num_sites, m, dtype=int,endpoint=False)
    effect_sizes = np.random.normal(loc=0, scale=(h2/m),size=m)
    
    all_effects = np.zeros(num_sites)
    np.put(all_effects,causal_inds,effect_sizes)
    return all_effects


def _calc_prs_tree(all_effects, h2, tree):
    non0 = np.where(all_effects!=0)[0]
    X_sum = 0
    i = 0
    for variant in tree.variants():
        if variant.site.id in non0:
            var_dip = return_diploid_genos(variant.genotypes,tree)
            X_sum+=np.dot(var_dip.T, all_effects[i])
        i+=1
    return X_sum

def _calc_prs_vcf(vcf_file,effects,n_admix):
    prs = np.zeros(n_admix)
    with open(vcf_file) as f:
        ind=0
        for line in f:
            if line[0] != "#":
                if effects[ind] != 0:
                    data = line.split("\t")[9:]
                    genotype = np.array([np.array(hap.split("|")).astype(int).sum() for hap in data])
                    prs=prs+(genotype*effects[ind])
                ind+=1
    return prs

def _summarize_true_prs(X,h2):
    Zx = (X-np.mean(X))/np.std(X)
    G = np.sqrt(h2)*Zx
    return X,Zx,G

def _write_true_prs(tree_ceu,tree_yri,m,h2,prs,effects,prefix):
    sample_ids_eur = ["ceu_{}".format(i) for i in range(0,int(tree_ceu.num_samples/2))]
    sample_ids_yri = ["yri_{}".format(i) for i in range(0,int(tree_yri.num_samples/2))]

    sample_ids_admix = list(pd.read_csv(prefix+"admixed_data/output/admix_afr_amer.prop.anc",sep="\t",index_col=0).index)
    sample_ids = np.concatenate((sample_ids_eur,sample_ids_yri,sample_ids_admix),axis=None)
    n_samps_ad = len(sample_ids_admix)
    n_all = len(sample_ids)

    X,Zx,G = summarize_true_prs(prs,h2)

    with h5py.File(prefix+"true_prs/prs_m_{}_h2_{}.hdf5".format(m,h2), 'w') as f:
        f.create_dataset("labels",(n_all,),data=sample_ids.astype("S"))
        f.create_dataset("X",(n_all,),dtype=float,data=X)
        f.create_dataset("Zx",(n_all,),dtype=float,data=Zx)
        f.create_dataset("G",(n_all,),dtype=float,data=G)
        f.create_dataset("effects",(n_sites,),dtype=float,data=effects)