import numpy as np, pandas as pd
import msprime,h5py
import gzip
import os
def simulate_true_prs(m,h2,n_admix,prefix="output/sim1/",
                        path_tree_CEU="trees/tree_CEU_GWAS_nofilt.hdf",
                        path_tree_YRI="trees/tree_YRI_GWAS_nofilt.hdf",
                        vcf_file="admixed_data/output/admix_afr_amer.query.vcf.gz"):
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
    n_admix : int
       number of admixed samples
    prefix : str, optional
        Output file path
    path_tree_CEU : str, optional
        Path to simulated tree containing CEU individuals
    path_tree_YRI : str, optional
        Path to simulated tree containing YRI individuals
    vcf_file : str, optional
        VCF file path with admixed genotypes

    """
    # Load simulation trees
    tree_CEU = msprime.load(prefix+path_tree_CEU)
    tree_YRI = msprime.load(prefix+path_tree_YRI)
    # Compute random effects for true PRS causal variants
    var_dict = _compute_effects(m, h2,tree_CEU.num_sites)
    # Calculate true PRS for Europeans and Africans
    prs_CEU = calc_prs_tree(var_dict,tree_CEU)
    prs_YRI = calc_prs_tree(var_dict,tree_YRI)
    # Calculate true PRS for admixed individuals
    prs_admix,samples_admix = calc_prs_vcf(prefix+vcf_file,var_dict,n_admix)
    # Combine all true PRS results
    prs_comb = np.concatenate((prs_CEU,prs_YRI,prs_admix),axis=None)
    # Write true PRS as hdf
    prs_norm, sample_ids = _write_true_prs(tree_CEU,tree_YRI,m,h2,
                                prs_comb,np.array(list(var_dict.values())),
                                prefix,samples_admix)
    # Get training and testing samples for creating and testing
    # the GWAS estimated PRS
    _split_case_control_all_pops(m,h2,prs_norm, sample_ids, n_admix,prefix)
    # For each admixed individual calculate the global proportion 
    # of European and African ancestry
    _calc_prop_total_anc(prefix)

    return

def return_diploid_genos(variant,tree):
    """
    Convert phased haplotypes to genotype

    Parameters
    ----------
    variant : tskit.Variant
        variant object from simulation tree
    tree : msprime.TreeSequence
        simulation tree

    Returns
    -------
    np.array
        genotypes for a given variant
    """
    genos_diploid = np.sum(variant.reshape([1,int(tree.num_samples/2),2]),axis=-1)
    return genos_diploid

def calc_prs_tree(var_dict,tree):
    """
    Create PRS from simulation tree for each individual
    for a given set of variants and weights

    Parameters
    ----------
    var_dict : dict
        keys - variant for PRS
        values - weight for PRS
    tree : msprime.TreeSequence
        simulation tree

    Returns
    -------
    int
        PRS for an individual
    """
    X_sum = 0 # Starting PRS value
    for variant in tree.variants(): # loop through variants in tree
        if variant.site.id in var_dict.keys(): # if variant in PRS
            var_dip = return_diploid_genos(variant.genotypes,tree) # get genotypes
            X_sum+=np.dot(var_dip.T, var_dict[variant.site.id]) # multiply by weights and sum
    return X_sum

def calc_prs_vcf(vcf_file,var_dict,n_admix):
    """
    Create PRS for admixed individuals
    for a given set of variants and weights

    Parameters
    ----------
    vcf_file : str
        VCF file path with admixed genotypes
    var_dict : dict
        keys - variant for PRS
        values - weight for PRS
    n_admix : int
        number of admixed samples

    Returns
    -------
    np.array
        PRS for each individual
    np.array
        admixed sample IDs
    """
    prs = np.zeros(n_admix) # initialize PRS
    with gzip.open(vcf_file,"rt") as f: # open VCF
        ind=0
        for line in f: # Get header for admixed sample IDs
            if line[:6]== "#CHROM":
                sample_ids_admix = line.split("\n")[0].split("\t")[9:]
            if line[0] != "#": # If line not header
                if ind in var_dict.keys(): # Check if variant in PRS
                    data = line.split("\n")[0].split("\t")[9:] # get phased haplotypes
                    # Convert to genotypes
                    genotype = np.array([np.array(hap.split("|")).astype(int).sum() for hap in data])
                    # Multiply by weight and add to PRS
                    prs=prs+(genotype*var_dict[ind])
                ind+=1
    return prs, sample_ids_admix

def _compute_effects(m, h2,num_sites):
    """
    Get random effect sizes for causal variants

    Parameters
    ----------
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    num_sites : int
        Total number of variants in genome

    Returns
    -------
    dict
        dictionary with variants and effects as 
        the keys and values
    """
    # select causal positions evently distributed throughout the genome
    causal_inds = np.linspace(0, num_sites, m, dtype=int,endpoint=False)
    # randomly assign effects
    effect_sizes = np.random.normal(loc=0, scale=(h2/m),size=m)
    # return dictionary with causal variant IDs and effects
    return dict(zip(causal_inds,effect_sizes))

def _summarize_true_prs(X,h2):
    """
    Output normalized and standardized PRS

    Parameters
    ----------
    X : np.array
        raw PRS across samples
    h2 : float
        Heritability due to genetics

    Returns
    -------
    int
        raw PRS across samples
    int
        normalized PRS
    int
        standardized PRS

    """ 
    # Mean center raw PRS
    Zx = (X-np.mean(X))/np.std(X)
    # Standardize to ensure total variance is h2
    G = np.sqrt(h2)*Zx
    return X,Zx,G

def _write_true_prs(tree_ceu,tree_yri,m,h2,prs,effects,prefix,sample_ids_admix):
    """
    Write PRS to file

    Parameters
    ----------
    tree_ceu : msprime.TreeSequence
        simulation tree Europeans
    tree_yri : msprime.TreeSequence
        simulation tree Africans
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    prs : np.array
        raw PRS across samples
    effects: np.array
        Something
    prefix : str
        Output file path
    sample_ids_admix : np.array
        Sample IDs for admixed individuals

    Returns
    -------
    np.array
        true PRS standardized to ensure total variance 
        is h2
    np.array
        sample IDs for Europeans, Africans, and admixed 
        individuals
    """
    # Create sample IDs for Europeans and Africans
    sample_ids_eur = ["ceu_{}".format(i) for i in range(0,int(tree_ceu.num_samples/2))]
    sample_ids_yri = ["yri_{}".format(i) for i in range(0,int(tree_yri.num_samples/2))]
    # Combine all sample IDs, including admixed
    sample_ids = np.concatenate((sample_ids_eur,sample_ids_yri,sample_ids_admix),axis=None)
    # Calculate number of admixed and total samples
    n_samps_ad = len(sample_ids_admix)
    n_all = len(sample_ids)
    # Get raw, normalized, and standardized true PRS
    X,Zx,G = _summarize_true_prs(prs,h2)
    # Write all data to hdf file
    with h5py.File(prefix+"true_prs/prs_m_{}_h2_{}.hdf5".format(m,h2), 'w') as f:
        f.create_dataset("labels",(n_all,),data=sample_ids.astype("S"))
        f.create_dataset("X",(n_all,),dtype=float,data=X)
        f.create_dataset("Zx",(n_all,),dtype=float,data=Zx)
        f.create_dataset("G",(n_all,),dtype=float,data=G)
        f.create_dataset("effects",effects.shape,dtype=float,data=effects)
    return G, sample_ids

def _split_case_control(G,E,inds,N_ADMIX):
    """
    Choose cases and controls for downstream GWAS

    Parameters
    ----------
    G : np.array
        true standardized PRS
    E : np.array
        enviornmental nosie to add to PRS
    inds : np.array
        Sample IDs
    N_ADMIX : int
        Number of admixed individuals

    Returns
    -------
    np.arrray
        training cases for GWAS
    np.array
        training controls for GWAS
    np.array
        testing samples
    """
    # Add PRS and enviornmental noise to get total trait liability
    G_plus_E = (G+E)[inds]
    # Assume 5% trait prevalence for cases
    num_case = int(len(inds)*0.05)
    # Get indices for sorted total liability
    sorted_risk = np.argsort(G_plus_E)[::-1]
    # Get sample ID for cases with top 5% trait liability
    train_case = inds[sorted_risk[:num_case]]
    # Subset to all samples except cases
    labels_control = inds[sorted_risk[num_case:]]
    # Randomly select an equal number of controls from
    # the remainder of the liability distribution
    train_controls = np.random.choice(labels_control, size=num_case, replace=False)
    remainder = [ind for ind in labels_control if ind not in train_controls]
    # Randomly select testing samples independent from the training samples
    # matching the number of admixed samples
    testing = np.random.choice(remainder, size=N_ADMIX, replace=False)
    return train_case,train_controls,testing

def _output_report_case_control(data):
    """
    Print details on training and testing split

    Parameters
    ----------
    data : list
        list containing cases, controls, and testing
        broken down by population
    """
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

def _split_case_control_all_pops(m,h2,G,labels,N_ADMIX,prefix):
    """
    Split Europeans and Africans into training (controls and cases)
    and testing samples

    Parameters
    ----------
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    G : np.array
        PRS for each sample
    labels : np.array
        Sample IDs
    N_ADMIX : int
        Number of admixed samples
    prefix : str, optional
        Output file path
    """

    # simulate random environmental noise to describe the remainder
    # of the heritability (1-h2)
    e = np.random.normal(loc=0, scale=(1-h2), size=int(G.shape[0])) # raw noise
    Ze = (e - np.mean(e))/np.std(e) # normalized to mean center 
    E = np.sqrt(1-h2)*Ze # standardized to have variance 1-h2
    # initialize arrays for sample IDs
    ceu_inds, yri_inds, admix_inds = [np.array([],dtype=int)]*3
    # get position of samples for each ancestry
    for ind,lab in enumerate(labels):
        if "ceu" in lab: ceu_inds = np.append(ceu_inds,[ind])
        elif "yri" in lab: yri_inds = np.append(yri_inds,[ind])
        else: admix_inds = np.append(admix_inds,[ind])
    # get cases, controls and testing for Europeans
    train_case_ceu,train_control_ceu,test_ceu = _split_case_control(G,E,ceu_inds,N_ADMIX)
    # get cases, controls and testing for Africans
    train_case_yri,train_control_yri,test_yri = _split_case_control(G,E,yri_inds,N_ADMIX)
    # combine all testing data
    test_w_admix = np.concatenate((test_ceu,test_yri,admix_inds),axis=None)
    # output summary of cases and controls
    _output_report_case_control([train_case_ceu,train_control_ceu,
                                train_case_yri,train_control_yri,
                                test_ceu,test_yri,admix_inds])
    # write data as hdf
    with h5py.File(prefix+'true_prs/prs_m_{}_h2_{}.hdf5'.format(m,h2), 'a') as f:
        f.create_dataset("train_cases_ceu",train_case_ceu.shape,dtype=int,data=train_case_ceu)
        f.create_dataset("train_controls_ceu",train_control_ceu.shape,dtype=int,data=train_control_ceu)
        f.create_dataset("train_cases_yri",train_case_yri.shape,dtype=int,data=train_case_yri)
        f.create_dataset("train_controls_yri",train_control_yri.shape,dtype=int,data=train_control_yri)
        f.create_dataset("test_data",test_w_admix.shape,dtype=int,data=test_w_admix)
        f.create_dataset("E",E.shape,dtype=float,data=E)

def _calc_prop_total_anc(prefix):
    """
    Get proportion of global European and African
    ancestry for each admixed individual

    Parameters
    ----------
    prefix : str, optional
        Output file path

    """
    # check if global ancestry already exists
    if not os.path.isfile(f"{prefix}admixed_data/output/admix_afr_amer.prop.anc"):
        # open file with phased haplotype ancestry at each variant for 
        # every admixed individual
        with gzip.open(f"{prefix}admixed_data/output/admix_afr_amer.result.gz","rt") as anc:
            print("Extracting proportion ancestry at PRS variants")
            for ind,line in enumerate(anc):
                # extract sample ids from header
                if ind == 0:
                    sample_haps = line.split("\t")[2:]
                    samples = [sample_haps[i].split(".")[0] for i in range(0,len(sample_haps),2)]
                    anc_prop = pd.DataFrame(index=samples,columns=["Prop_CEU","Prop_YRI"])
                    counts_YRI = np.zeros(len(samples)) # initialize ancestry counts African
                    counts_CEU = np.zeros(len(samples)) # inditialize ancestry counts European
                # check ancestry at each variant
                else:
                    # get haplotype ancestry
                    # array of 1s and 2s
                    # 1 = European, 2 = African
                    haplo_anc = np.array(line.split("\t")[2:]).astype(int)
                    # get binary array for African ancestry  
                    YRI_arr = haplo_anc-1
                    # convert haploid ancestry to diploid value
                    line_counts_YRI = np.add.reduceat(YRI_arr, np.arange(0, len(YRI_arr), 2))
                    # get binary array for European ancestry
                    CEU_arr = np.absolute(1-YRI_arr)
                    # convert haploid ancestry to diploid value
                    line_counts_CEU = np.add.reduceat(CEU_arr, np.arange(0, len(CEU_arr), 2))
                    # get counts of each ancestry for every individual
                    # at a particular position an individual can have 0, 1, or 2 African alleles
                    # or 0, 1, or 2 European alleles
                    counts_YRI = counts_YRI+line_counts_YRI
                    counts_CEU = counts_CEU+line_counts_CEU
        # Calculate global ancestry proportions
        anc_prop["Prop_YRI"] = counts_YRI/(2*(ind-1))
        anc_prop["Prop_CEU"] = counts_CEU/(2*(ind-1))
        # Write to file
        anc_prop.to_csv(f"{prefix}admixed_data/output/admix_afr_amer.prop.anc",sep="\t")
    return
