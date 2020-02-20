import numpy as np, pandas as pd
import msprime,h5py

def simulate_true_prs(m,h2,n_admix,prefix="output/sim1/",
                        path_tree_CEU="trees/tree_CEU_GWAS_nofilt.hdf",
                        path_tree_YRI="trees/tree_YRI_GWAS_nofilt.hdf",
                        vcf_file="admixed_data/output/admix_afr_amer.query.vcf"):
    """
    Functions to create true polygenic risk score for simulated data and 
    select cases+controls in the data.
    
    True PRS is constructed as defined in [Martin et. al. 2017 AJHG](https://www.ncbi.nlm.nih.gov/pubmed/28366442).
    In short, m causal variants that are evenly spaced across the genome and selected
    and random effects are generated constrained by the total heritability.
    
    Environmental noise is added to make up the total liability where
    5% disease prevalence is assumed to select cases. Controls are randomly
    selected from the remainder of the distribution.

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
    tree_CEU = msprime.load(prefix+path_tree_CEU)
    tree_YRI = msprime.load(prefix+path_tree_YRI)
    all_effects = _compute_effects(m, h2,tree_CEU.num_sites)
    
    prs_CEU = _calc_prs_tree(all_effects,h2,tree_CEU)
    prs_YRI = _calc_prs_tree(all_effects,h2,tree_YRI)
    prs_admix,samples_admix = _calc_prs_vcf(prefix+vcf_file,all_effects,n_admix)
    prs_comb = np.concatenate((prs_CEU,prs_YRI,prs_admix),axis=None)

    prs_norm, sample_ids = _write_true_prs(tree_CEU,tree_YRI,m,h2,prs_comb,all_effects,prefix,samples_admix)
    _split_case_control_all_pops(h2,prs_norm, sample_ids, n_admix)

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
            var_dip = _return_diploid_genos(variant.genotypes,tree)
            X_sum+=np.dot(var_dip.T, all_effects[i])
        i+=1
    return X_sum

def _calc_prs_vcf(vcf_file,effects,n_admix):
    prs = np.zeros(n_admix)
    with open(vcf_file) as f:
        ind=0
        for line in f:
            if line[:6]== "#CHROM":
                sample_ids_admix = line.split("\n")[0].split("\t")[9:]
            if line[0] != "#":
                if effects[ind] != 0:
                    data = line.split("\t")[9:]
                    genotype = np.array([np.array(hap.split("|")).astype(int).sum() for hap in data])
                    prs=prs+(genotype*effects[ind])
                ind+=1
    return prs, sample_ids_admix

def _summarize_true_prs(X,h2):
    Zx = (X-np.mean(X))/np.std(X)
    G = np.sqrt(h2)*Zx
    return X,Zx,G

def _write_true_prs(tree_ceu,tree_yri,m,h2,prs,effects,prefix,sample_ids_admix):
    sample_ids_eur = ["ceu_{}".format(i) for i in range(0,int(tree_ceu.num_samples/2))]
    sample_ids_yri = ["yri_{}".format(i) for i in range(0,int(tree_yri.num_samples/2))]

    sample_ids = np.concatenate((sample_ids_eur,sample_ids_yri,sample_ids_admix),axis=None)
    n_samps_ad = len(sample_ids_admix)
    n_all = len(sample_ids)

    X,Zx,G = _summarize_true_prs(prs,h2)

    with h5py.File(prefix+"true_prs/prs_m_{}_h2_{}.hdf5".format(m,h2), 'w') as f:
        f.create_dataset("labels",(n_all,),data=sample_ids.astype("S"))
        f.create_dataset("X",(n_all,),dtype=float,data=X)
        f.create_dataset("Zx",(n_all,),dtype=float,data=Zx)
        f.create_dataset("G",(n_all,),dtype=float,data=G)
        f.create_dataset("effects",(tree_ceu.num_sites,),dtype=float,data=effects)
    return G, sample_ids

def _split_case_control(G,E,inds,N_ADMIX):
    G_plus_E = (G+E)[inds]
    num_case = int(len(inds)*0.05)
    sorted_risk = np.argsort(G_plus_E)[::-1]
    train_case = inds[sorted_risk[:num_case]]
    labels_control = inds[sorted_risk[num_case:]]
    train_controls = np.random.choice(labels_control, size=num_case, replace=False)
    remainder = [ind for ind in labels_control if ind not in train_controls]
    testing = np.random.choice(remainder, size=N_ADMIX, replace=False)
    return train_case,train_controls,testing

def _output_report_case_control(data):
    print("-----------------------------")
    print("Sample Training/Testing Split")
    print("-----------------------------")
    print("Cases CEU = {}".format(len(data[0])))
    print("Cases YRI = {}".format(len(data[1])))
    print("Controls CEU = {}".format(len(data[2])))
    print("Controls YRI = {}".format(len(data[3])))
    print("Testing CEU = {}".format(len(data[4])))
    print("Testing YRI = {}".format(len(data[5])))
    print("Testing admixed (CEU+YRI) = {}".format(len(data[6])))
    print("-----------------------------")

def _split_case_control_all_pops(h2,G,labels,N_ADMIX):
    e = np.random.normal(loc=0, scale=(1-h2), size=int(G.shape[0]))
    Ze = (e - np.mean(e))/np.std(e)
    E = np.sqrt(1-h2)*Ze
    ceu_inds, yri_inds, admix_inds = [np.array([],dtype=int)]*3
    
    for ind,lab in enumerate(labels):
        if "ceu" in lab: ceu_inds = np.append(ceu_inds,[ind])
        elif "yri" in lab: yri_inds = np.append(yri_inds,[ind])
        else: admix_inds = np.append(admix_inds,[ind])
    
    train_case_ceu,train_control_ceu,test_ceu = _split_case_control(G,E,ceu_inds,N_ADMIX)
    train_case_yri,train_control_yri,test_yri = _split_case_control(G,E,yri_inds,N_ADMIX)
    test_w_admix = np.concatenate((test_ceu,test_yri,admix_inds),axis=None)

    _output_report_case_control([train_case_ceu,train_control_ceu,
                                train_case_yri,train_control_yri,
                                test_ceu,test_yri,admix_inds])


