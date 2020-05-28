"""
Functions to calculate a empirical PRS for
each population given a snp weighing and snp 
selection approach.

Description of options:
snp selection - population, p-value, LD r2, sample size
snp weighting - populations
"""

import numpy as np, pandas as pd, math
np.seterr(divide='ignore', invalid='ignore')
import msprime
import sys, threading
import gzip, h5py, os
from scipy import stats
import statsmodels.api as sm
import tqdm
import gzip
from scipy import stats
from scipy.stats import chi2

import matplotlib.pyplot as plt
import seaborn as sns

from .true_risk import return_diploid_genos
from .true_risk import calc_prs_tree, calc_prs_vcf

def create_emp_prs(m,h2,n_admix,prefix,p,r2,
    vcf_file = "admixed_data/output/admix_afr_amer.query.vcf.gz",
    path_tree_CEU="trees/tree_CEU_GWAS_nofilt.hdf",
    path_tree_YRI="trees/tree_YRI_GWAS_nofilt.hdf",
    snp_weighting="ceu",snp_selection="ceu",
    num2decrease=None,ld_distance=1e6,num_threads=8):
    """
    This function carries out all steps required to 
    create a PRS for Europeans, Africans, and 
    admixed individuals given a selected PRS SNP 
    selection and weighted strategy

    Parameters
    ----------
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    n_admix : msprime.simulations.RecombinationMap
        Recombination map for a reference chromosome
    prefix : str
        Output file path
    p : float
        P-value to use for summary statistic thresholding
    r2 : float
        LD r2 to use for summary statistic clumping
    vcf_file : str, optional
        VCF file path with admixed genotypes
    path_tree_CEU : str, optional
        Path to simulated tree containing CEU individuals
    path_tree_YRI : str, optional
        Path to simulated tree containing YRI individuals
    snp_weighting: str, optional
        Population to use for SNP weights
    snp_selection: str, optional
        Population to use for SNP selection
    num2decrease: int, optional
        Number of non-European samples to use in GWAS
    ld_distance: int, optional
        LD window size to use for clumping
    num_threads: int, optional
        number of threads to use for parallel processing
    """

    # get simulation number
    sim = prefix.split('sim')[1].split('/')[0]
    # load all data
    trees,sumstats,train_cases,train_controls,labels = _load_data(snp_weighting,snp_selection,
    															  path_tree_CEU,
                                                                  path_tree_YRI,prefix,m,h2,
                                                                  num2decrease)
    # select PRS variants given above parameters
    snps = _select_variants(sumstats[snp_selection],trees[snp_selection],m,h2,
                           p,r2,snp_selection,prefix,ld_distance,num_threads,num2decrease)
    
    # compare causal SNP allele frequencies to GWAS selected SNP allele frequencies
    causal_inds =  np.linspace(0, trees["ceu"].num_sites, m, dtype=int,endpoint=False)
    _write_allele_freq_bins(sim,causal_inds,trees,prefix,snp_selection,prefix+vcf_file,
    						m,h2,r2,p,train_cases,causal=True)
    _write_allele_freq_bins(sim,snps.astype(int),trees,prefix,snp_selection,prefix+vcf_file,
    						m,h2,r2,p,train_cases) 

    # if empirical PRS already exists output warning
    if os.path.isfile(f"{prefix}emp_prs/emp_prs_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{snp_selection}_"+\
                      f"snps_{len(train_cases[snp_selection])}cases_{snp_weighting}_weights_"+\
                      f"{len(train_cases[snp_weighting])}cases.hdf5"):
        print(f"\nEmpirical PRS for iteration={prefix.split('sim')[1].split('/')[0]}"+\
        		" and provided parameters exists")
        print(f"If you would like to overwrite, remove {prefix}emp_prs/emp_prs_m_{m}_h2"+\
        		f"_{h2}_r2_{r2}_p_{p}_{snp_selection}_snps_{len(train_cases[snp_selection])}cases"+\
        		f"_{snp_weighting}_weights_{len(train_cases[snp_weighting])}cases.hdf5")
        return
    
    # if empirical PRS does not exist, calculate for all populations
    else:
        print(f"Creating empricial with {snp_selection.upper()} selected snps and {snp_weighting.upper()} weights ")
        # SNP weights are not by local ancestry, create PRS by using weights from a single or meta study
        if snp_weighting != "la":
            # convert ORs to effect sizes, missing SNPs will have effect size of 0
            weights = np.log(sumstats[snp_weighting].reindex(snps,fill_value=1)["OR"])
            print("..... for the CEU population")
            # caculate PRS in Europeans
            prs_ceu = calc_prs_tree(dict(zip(snps,weights)),trees["ceu"])
            print("..... for the YRI population")
            # calculate PRS in Africans
            prs_yri = calc_prs_tree(dict(zip(snps,weights)),trees["yri"])
            # calculate PRS in admixed population
            print("..... for the admixed population")
            prs_admix,ids_admix = calc_prs_vcf(prefix+vcf_file,dict(zip(snps,weights)),n_admix)
            # combine all PRS
            prs_all = np.concatenate((prs_ceu,prs_yri,prs_admix),axis=None)
            # write output
            _write_output(prs_all,labels,prefix,m,h2,r2,p,snp_selection,snp_weighting,
                    len(train_cases[snp_weighting]),len(train_cases[snp_selection]))
        # if SNP weights are by local ancestry
        else:
            # write out the ancestry for each PRS SNP
            _ancestry_snps_admix(snps,prefix,m,h2,r2,p,snp_selection,num2decrease)
            # create data dictionary of all study weights
            weights = {"ceu":np.log(sumstats["ceu"].reindex(snps,fill_value=1)["OR"]),
                       "yri":np.log(sumstats["yri"].reindex(snps,fill_value=1)["OR"]),
                       "meta":np.log(sumstats["meta"].reindex(snps,fill_value=1)["OR"])}
            # calculate PRS in Europeans (100% European ancestry)
            print("..... for the CEU population")
            prs_ceu = calc_prs_tree(dict(zip(snps,weights["ceu"])),trees["ceu"])
            # calculate PRS in Africans (100% African ancestry)
            print("..... for the YRI population")
            prs_yri = calc_prs_tree(dict(zip(snps,weights["yri"])),trees["yri"])
            # calculate PRS in admixed population, weight varies by local ancestry
            print("..... for the admixed population")
            prs_admix = calc_prs_vcf_la(prefix+vcf_file,weights,snps,n_admix,m,h2,r2,p,
                                        snp_selection,prefix,trees["ceu"].num_sites,num2decrease)
            # Ccombine all results
            prs_all = np.concatenate((prs_ceu,prs_yri,prs_admix),axis=None)
            # write output
            _write_output(prs_all,labels,prefix,m,h2,r2,p,snp_selection,snp_weighting,
                    len(train_cases[snp_weighting]),len(train_cases[snp_selection]))
            return
    return

def calc_prs_vcf_la(vcf_file,weights,snps,n_admix,m,h2,r2,
                    p,pop,prefix,num_sites,num2decrease):
    """
    Function to calculate local ancestry weighted PRS for
    admixed individuals

    Parameters
    ----------
    vcf_file : str
        VCF file path with admixed genotypes
    weights : dict
        summary statistics for each ancestry population
        key - population code
        value - summary statistics
    snps : list
        variant ids included in the PRS
    n_admix : int
        number of admixed samples
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    r2 : float
        LD r2 used for clumping
    p : float
        p-value used for thresholding
    pop : str
        population used for SNP selection
    prefix : str
        Output file path
    num_sites : int
        Number of sites in the genome
    num2decrease: int
        Number of non-European samples to use in GWAS

    Returns
    -------
    numpy.array
        PRS for each individual
    """

    # load ancestry for each PRS variant
    if num2decrease==None: # number of African samples match Europeans
        anc_file = f"{prefix}admixed_data/output/admix_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{pop}_snps.result.PRS"
    else: # number of African samples are smaller than Europeans
        anc_file = f"{prefix}admixed_data/output/admix_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{pop}_snps_{num2decrease}.result.PRS"
    anc = pd.read_csv(anc_file,sep="\t",index_col=0) # ancestry for PRS variants
    # get sample IDs for admixed samples
    sample_ids = pd.read_csv(f"{prefix}admixed_data/output/admix_afr_amer.prop.anc",
        sep="\t",index_col=0).index
    # initialize PRS vector
    prs = np.zeros(n_admix)
    # output progress
    pbar = tqdm.tqdm(total=num_sites)
    # open admix vcf file with genotypes of PRS variants
    with gzip.open(vcf_file,"rt") as f:
        ind=0 # track position in file
        for line in f: # loop through file lines
            if line[0] != "#": # ignore header
                if ind in snps: # loop through SNPs
                    data = line.split("\t")[9:] # extract phased haplotypes from vcf line
                    # convert to genotype
                    genotype = np.array([np.array(hap.split("|")).astype(int).sum() for hap in data])
                    var_weighted=[] # get weighted genotype for each PRS variant
                    for g in range(0,len(genotype)): # loop through individuals
                        # check individual ancestry and weight accordingly
                        # if individual is diploid African, use African GWAS weight 
                        if anc.loc[ind,[sample_ids[g]+".0",sample_ids[g]+".1"]].sum() > 3:
                            var_weighted.append(genotype[g]*weights["yri"][ind])
                        # if individual is haploid European/African, use Meta GWAS weight
                        elif anc.loc[ind,[sample_ids[g]+".0",sample_ids[g]+".1"]].sum() == 3:
                            var_weighted.append(genotype[g]*weights["meta"][ind])
                        # if individual is diploid European, use European GWAS weight
                        elif anc.loc[ind,[sample_ids[g]+".0",sample_ids[g]+".1"]].sum() < 3:
                            var_weighted.append(genotype[g]*weights["ceu"][ind])
                    # add weighted variant for each person to overall PRS
                    prs=prs+np.array(var_weighted)
                ind+=1 # update position of variant
                pbar.update(1) # update progress bar
    return prs

def _ancestry_snps_admix(snps,prefix,m,h2,r2,p,pop,num2decrease):
    """
    Function to extract ancestry at PRS variants for admixed population

    Parameters
    ----------
    snps : list
        variant ids included in the PRS
    prefix : str
        Output file path
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    r2 : float
        LD r2 used for clumping
    p : float
        p-value used for thresholding
    pop : str
        population used for SNP selection
    num2decrease: int
        Number of non-European samples to use in GWAS
    """

    # load admixed individuals
    if num2decrease==None: 
        outfile = f"{prefix}admixed_data/output/admix_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{pop}_snps.result.PRS"
    else: 
        outfile = f"{prefix}admixed_data/output/admix_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{pop}_snps_{num2decrease}.result.PRS"
    # if ancestry for PRS variants does not exist extract from genome-wide ancestry
    if not os.path.isfile(outfile):
        with gzip.open(f"{prefix}admixed_data/output/admix_afr_amer.result.gz","rt") as anc:
            print("Extracting proportion ancestry at PRS variants")
            for ind,line in enumerate(anc):
                if ind == 0: # extract header
                    anc_prs = pd.DataFrame(columns=line.split("\n")[0].split("\t")[2:])

                elif ind-1 in snps: # check if SNP in PRS and if it is get phased ancestry
                    anc_prs.loc[ind-1,:] = line.split("\n")[0].split("\t")[2:]
        # write ancestry to file
        anc_prs.to_csv(outfile,sep="\t")
    return

def _perform_meta(train_cases,m,h2,prefix):
    """
    Function to perform fixed-effects meta from European and 
    African summary statistics

    Parameters
    ----------
    train_cases : dict
        samples used for training
        key - population
        value - sample IDs
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    prefix : str
        Output file path

    Returns
    -------
    pd.DataFrame
        fixed-effects meta summary statistics
    """
    # check if meta file already exists and if not create it
    if not os.path.isfile(prefix+f"emp_prs/meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases['ceu'])}"+ \
                                 f"_casesYRI_{len(train_cases['yri'])}.txt"):
        # call Rscript to perform meta analysis
        print("\nPerforming a fixed_effects meta between CEU and YRI summary statistics")
        os.system("Rscript simulation/compute_meta_sum_stats.R " + \
                 f"{prefix}emp_prs/gwas_m_{m}_h2_{h2}_pop_ceu_cases_{len(train_cases['ceu'])}.txt " + \
                 f"{prefix}emp_prs/gwas_m_{m}_h2_{h2}_pop_yri_cases_{len(train_cases['yri'])}.txt " + \
                 f"{prefix}emp_prs/meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases['ceu'])}_casesYRI_{len(train_cases['yri'])}.txt")
        sum_stats = pd.read_csv(prefix+f"emp_prs/meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases['ceu'])}"+ \
                                 f"_casesYRI_{len(train_cases['yri'])}.txt",sep="\t",index_col=0)
        # plot QQ plot of the results
        _plot_qq(sum_stats,prefix,f"meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases['ceu'])}_casesYRI_{len(train_cases['yri'])}")

    return pd.read_csv(prefix+f"emp_prs/meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases['ceu'])}_casesYRI_{len(train_cases['yri'])}.txt",sep="\t",index_col=0)

def _select_variants(sum_stats,tree,m,h2,p,r2,pop,prefix,
    max_distance,num_threads,num2decrease):
    """
    Function to select PRS variants

    Parameters
    ----------
    sum_stats : pd.DataFrame
        GWAS summary statistics for a population
    tree : msprime.TreeSequence
        population tree created from msprime
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    p : float
        p-value used for thresholding
    r2 : float
        LD r2 used for clumping
    pop : str
        population used for SNP selection
    prefix : str
        Output file path
    max_distance : int
        LD window size
    num_threads: int
        number of threads to use for parallel processing
    num2decrease: int
        Number of non-European samples to use in GWAS

    Returns
    -------
    numpy.array
        PRS variant IDs
    """
    print("-----------------------------------")
    print("Selecting variants for PRS building")
    print("-----------------------------------")
    print(f"Population used for LD clumping = {pop}")
    print(f"Parameters: p-value = {p} and r2 = {r2}")
    # get thresholded variants by p-value
    prs_vars = sum_stats[sum_stats["p-value"] < p].sort_values(by=["p-value"]).index
    print(f"# variants with p < {p}: {len(prs_vars)}")
    # clump variants
    clumped_prs_vars = _ld_clump(tree,prs_vars,m,h2,pop,r2,p,
                                 prefix,max_distance,num_threads,num2decrease)
    print(f"# variants after clumping: {len(clumped_prs_vars)}")
    print("-----------------------------------")
    # return significant independent variant list
    return clumped_prs_vars

def _compute_maf_vcf(vcf_file,var_list):
    """
    Calculate MAF for admixed individuals

    Parameters
    ----------
    vcf_file : str
        VCF file path with admixed genotypes
    var_list : np.array
        list of variants included in the PRS

    Returns
    -------
    numpy.array
        PRS variant minor allele frequencies
    """
    mafs = []
    # loop through vcf file
    with gzip.open(vcf_file,"rt") as f:
        ind = 0
        for line in f:
            if line[0] != "#":
                # if variants in PRS, calculate MAF
                if ind in var_list:
                    # get phased haplotypes
                    data = line.split("\n")[0].split("\t")[9:]
                    # convert to genotype
                    genotype = np.array([np.array(hap.split("|")).astype(int).sum() for hap in data])
                    genotype = genotype.reshape(1,len(genotype))
                    # calculate allele frequency
                    freq = np.sum(genotype,axis=1)/(2*genotype.shape[1])
                    # calculate MAF
                    if freq < 0.5: maf = freq
                    else: maf = 1-freq
                    mafs.append(maf)
                ind+=1

    return np.array(mafs)

def _return_maf_group(mafs,n_sites):
    """
    Helper function for plotting
    minor allele freuqncy (MAF) distribution
    of PRS variants

    Parameters
    ----------
    mafs : np.array
        Minor allele frequency for variants
    n_sites : int
        Total number of variants

    Returns
    -------
    list
        Percent of sites in each MAF group
    """
    G1 = len(mafs[mafs<0.01])/n_sites
    G2 = len(mafs[(mafs>=0.01)&(mafs<0.1)])/n_sites
    G3 = len(mafs[(mafs>=0.1)&(mafs<0.2)])/n_sites
    G4 = len(mafs[(mafs>=0.2)&(mafs<0.3)])/n_sites
    G5 = len(mafs[(mafs>=0.3)&(mafs<0.4)])/n_sites
    G6 = len(mafs[(mafs>=0.4)&(mafs<0.5)])/n_sites
    return [G1,G2,G3,G4,G5,G6]

def _write_allele_freq_bins(sim,var_list,trees,prefix,snp_selection,vcf_file,
    m,h2,r2,p,train_cases,causal=False):
    """
    Helper function to write MAF bins for list of variants.
    MAF calculated within each population.

    Parameters
    ----------
    sim : int
        simulation identifier
    var_list : np.array
        Variants to compute AF for
    trees : dict
        key - population
        value - msprime.TreeSequence
    prefix : str
        Output file path
    snp_selection: str
        Population to use for SNP selection
    vcf_file : str
        VCF file path with admixed genotypes
    """
    # check if MAF has already been calculated
    if not os.path.isfile(f"{prefix}summary/emp_maf_bins_m_{m}_h2_{h2}_r2_{r2}"+\
    					  f"_p_{p}_{snp_selection}_snps_{len(train_cases)}cases"+\
    					  f"_{sim}.txt"):
        # compute MAF in Europeans
        maf_ceu = _compute_maf(trees["ceu"],prefix,"ceu")[var_list]
        # compute MAF in Africans
        maf_yri = _compute_maf(trees["yri"],prefix,"yri")[var_list]
        # compute MAF in admixed individuals
        maf_admix = _compute_maf_vcf(vcf_file,var_list)
        # get proportion of variants that fall in each MAF bin
        bins = ["0 - 0.01","0.01 - 0.1","0.1 - 0.2","0.2 - 0.3","0.3 - 0.4","0.4 - 0.5"]
        df = pd.DataFrame(columns=["sim","pop"]+bins)
        for pop,mafs in zip(["ceu","yri","admix"],[maf_ceu,maf_yri,maf_admix]):
            groups = _return_maf_group(mafs,len(mafs))
            sub_df = pd.DataFrame([[sim,pop]+groups],columns=["sim","pop"]+bins)
            df = df.append(sub_df, ignore_index=True)
        # write files
        if not causal:
            df.to_csv(f"{prefix}summary/emp_maf_bins_m_{m}_h2_{h2}_r2_{r2}_p_{p}"+\
                        f"_{snp_selection}_snps_{len(train_cases[snp_selection])}cases_"+\
                        f"{sim}.txt",sep="\t")
        else:
            df.to_csv(f"{prefix}summary/causal_maf_bins_m_{m}_h2_{h2}_{sim}.txt",sep="\t")
    return

def _compute_summary_stats(m,h2,tree,train_cases,train_controls,pop,prefix):
	"""
    This function computes the summary statistics for
    each variant in the genome if it has maf > 1%

    Parameters
    ----------
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    tree : msprime.TreeSequence
        simulation tree with genotypes
    train_cases : np.array
    	cases used in GWAS
    train_controls : np.array
    	controls used in GWAS
    pop : str
    	Population used to calculate LD
    prefix : str
        Output file path

    Return
    ------
    pd.DataFrame
    	genome-wide summary statistics
    """
    # ouptut file name
    outfile = f"{prefix}emp_prs/gwas_m_{m}_h2_{h2}_pop_{pop}_cases_{len(train_cases)}.txt"
    # check if output file exists
    if not os.path.isfile(outfile):
        print(f"Computing SNP MAF for {pop.upper()}. GWAS will be performed for SNPs with maf > 1%")
        # calculate minor allele frequencies
        mafs = _compute_maf(tree,prefix,pop)
        # exclude variants with minor allele frequency < 1%
        var_ids = np.where(np.array(mafs) >= 0.01)[0]
        # print cases and sites used in GWAS
        print("Running GWAS for population = {}".format(pop.upper()))
        print("------------------- # cases = {}".format(len(train_cases)))
        print("------------------- # sites = {}".format(len(var_ids)))
        print("\n")
        # initialize summary statistics arrray
        sum_stats_arr = np.empty(shape=(len(var_ids),3))
        pbar = tqdm.tqdm(total=tree.num_sites) # track progress
        
        var_loc = 0 # track current variant
        for var in tree.variants():
            if var.site.id in var_ids: # check that the variant has maf >= 1%
            	# convert phased haplotypes to genotypes
                genos = return_diploid_genos(var.genotypes,tree)
                # get genotypes for cases
                genos_cases = genos[:,train_cases]
                # get genotypes for controls
                genos_controls = genos[:,train_controls]
                # calculate odds ratio and p-value
                OR,pval = _gwas(genos_cases,genos_controls)
                # add result to summary statistics
                sum_stats_arr[var_loc]=[var.site.id,OR,pval]
                var_loc+=1
            pbar.update(1) # update progress bar
        # convert results to a dataframe
        sum_stats = pd.DataFrame(sum_stats_arr,columns=["var_id","OR","p-value"])
        # replace infinite values with NA
        sum_stats = sum_stats.replace([np.inf, -np.inf], np.nan)
        # drop snps that could not be computed
        sum_stats.dropna(inplace=True)
        # set index to variant ID
        sum_stats = sum_stats.set_index("var_id").sort_index()
        # write output file
        sum_stats.to_csv(outfile,sep="\t",index=True)
        # make QQ plot of results
        _plot_qq(sum_stats,prefix,f"gwas_m_{m}_h2_{h2}_pop_{pop}_cases_{len(train_cases)}")
        return sum_stats
    else: 
        sum_stats = pd.read_csv(outfile,sep="\t",index_col=0)
        return sum_stats

def _plot_qq(sum_stats,prefix,outfile):
    """
    Create a QQ plot from GWAS summary statistics

    Parameters
    ----------
    sum_stats : pd.DataFrame
    	odds ratios and p-values for variants
    prefix : str
        Output file path
    outfile : str
        filename
    """
    # calculate lambda gc
    chisq = chi2.ppf(1-sum_stats["p-value"],1)
    lam_gc = np.median(chisq)/chi2.ppf(0.5,1)
    # observed p-values
    pvals = sum_stats["p-value"].values
    # expected p-values
    expected_p = (stats.rankdata(pvals,method="ordinal")+0.5)/(len(pvals)+1)
    # plot and save figure
    plt.figure(figsize=(5,5))
    plt.scatter(-1*np.log10(expected_p), -1*np.log10(pvals),s=20)
    plt.plot(sorted(-1*np.log10(expected_p)),sorted(-1*np.log10(expected_p)),c="black",linestyle="--")
    plt.text(2.5,200,"$\lambda$ = {}".format(np.round(lam_gc,2)),fontsize=20)
    plt.xlabel("Expected -log10 P-value",fontsize=16)
    plt.ylabel("Observed -log10 P-Value",fontsize=16)
    plt.ylim(0,300)
    sns.despine()
    plt.savefig(f"{prefix}summary/{outfile}_QQ.png",type="png",bbox_inches="tight",dpi=400)



def _ld_clump(tree,variants,m,h2,pop,r2,p,prefix,max_distance,num_threads,num2decrease):
    """
    Function to perform LD clumping for candidate PRS
    variants with p-value less than a supplied threshold

    Parameters
    ----------
    tree : msprime.TreeSequence
        simulation tree with population genotypes
    variants : np.array
        variants to check for LD, ordered by p-value
        most significant variants at beginning
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    pop : str
    	Population used to calculate LD
    r2 : float
        LD r2 to use for summary statistic clumping
    p : float
        P-value to use for summary statistic thresholding  
    prefix : str
        Output file path
    max_distance: int
        LD window size to use for clumping
    num_threads: int
        number of threads to use for parallel processing
    num2decrease: int
        Number of non-European samples to use in GWAS
    """
    # set file path to contain clumped prs variants
    if num2decrease == None: 
    	path = prefix+f"emp_prs/clumped_prs_vars_m_{m}_h2_{h2}_pop_{pop}_r2_{r2}_p{p}.txt"
    else: 
    	path = prefix+f"emp_prs/clumped_prs_vars_m_{m}_h2_{h2}_pop_{pop}_r2_{r2}_p{p}_cases_{num2decrease}.txt"
    # check if clumped variants already exist
    if not os.path.isfile(path): # if file does not exist
        print("Clumping variants...")
        var2mut,mut2var = _get_var_mut_maps(tree) # dictionaries for variants to mutation relationship
        tree_ld = tree.simplify(filter_sites=True) # filter missing variants from tree to compute LD
        # get variants with high LD within a set window size for candidate PRS variants
        ld_struct = _compute_ld_variants(tree_ld,variants,r2,var2mut,mut2var,max_distance,num_threads)
        # add most significant variant to clumped list
        clumped_variants = [variants[0]]
        # for remaining variants check if they are in LD with current variants in clumped list
        for v in range(1,len(variants)):
            add, i = True, 0 # initialize boolean and position in clumped list
            # continue checking LD until you find high r2 with variant in clumped list
            while add and i < len(clumped_variants):
            	# if variant in LD do not add it
                if variants[v] in ld_struct.get(clumped_variants[i]):
                    add = False
                # check next clumped variant
                i+=1
             # if after checking all variants, there is no high LD then added to clumped list
            if add: clumped_variants.append(variants[v])
        # save clumped variants to file and return
        np.savetxt(path,clumped_variants)
        return np.array(clumped_variants)
    # if clumped variants already exist load file and return 
    else: 
        return np.loadtxt(path)

def _get_var_mut_maps(tree):
    """
    Get mutation-variant mappings from
    simulation tree

    Parameters
    ----------
    tree : msprime.TreeSequence
        simulation tree with population genotypes

    Returns
    -------
    dict
    	associated mutation with each variant
        key - variant
        value - mutation
        See msprime document for details.
    dict
        associated variant with each mutation
        key - mutation
        value - variant
        See msprime document for details.
    """
    var2mut, mut2var = {}, {}
    for mut in tree.mutations():
        mut2var[mut.id]=mut.site
        var2mut[mut.site]=mut.id
    return var2mut, mut2var

def _compute_ld_variants(tree,focal_vars,r2,var2mut_dict,
            mut2var_dict,max_distance,num_threads):
    """
    Identify variants with LD r2 greater than a defined
    threshold for each focal variant 
    (i.e. each candidate PRS variant before clumping)

    Parameters
    ----------
    tree : msprime.TreeSequence
        simulation tree with population genotypes
    focal_vars : np.array
        variants to check for LD
    r2 : float
        LD r2 threshold to use for clumping
    var2mut_dict : dict
        associated mutation with each variant
        key - variant
        value - mutation
        See msprime document for details.
    mut2var_dict : dict
        associated variant with each mutation
        key - mutation
        value - variant
        See msprime document for details.
    max_distance : int
        LD window size to use for clumping
    num_threads: int
        number of threads to use for parallel processing

    Returns
    -------
    dict
    	variants with LD r2 greater than the threshold
		for each focal variant
		key - focal variant
		value - list of variants in LD with focal variant
    """
    results = {} # initialize LD dictionary
    num_threads = min(num_threads, len(focal_vars)) # initialize threads

    # distribute variants across threads to check for LD
    def thread_worker(thread_index):
    	# initialize LD calculator
        ld_calc = msprime.LdCalculator(tree)
        # split variants into chunks that can be shared across threads
        chunk_size = int(math.ceil(len(focal_vars) / num_threads))
        start = thread_index * chunk_size
        # for each focal variant find variants within a window size
        # that have LD r2 greater than a threshold
        for focal_var in focal_vars[start: start + chunk_size]:
        	# convert variant to mutation for msprime LD calculation
            focal_mutation = var2mut_dict.get(focal_var)
            np.seterr(under='ignore')
            # get LD for variants to the left of focal variant
            a = ld_calc.get_r2_array(
                focal_mutation, max_distance=max_distance,
                direction=msprime.REVERSE)
            # if LD not present set to zero (as specified by msprime docs)
            a[np.isnan(a)] = 0
            # get variant positions with high LD
            rev_indexes = focal_mutation - np.nonzero(a >= r2)[0] - 1
            # get LD for variants to the right of focal variant
            a = ld_calc.get_r2_array(
                focal_mutation, max_distance=max_distance,
                direction=msprime.FORWARD)
            # if LD not present set to zero (as specified by msprime docs)
            a[np.isnan(a)] = 0
            # get variant positions with high LD
            fwd_indexes = focal_mutation + np.nonzero(a >= r2)[0] + 1
            # combine variants to the left and right of a focal variant with high LD
            indexes = np.concatenate((rev_indexes[::-1], fwd_indexes))
            # convert mutation back to variant
            indexes = [mut2var_dict.get(ind) for ind in indexes if mut2var_dict.get(ind)!=None]
            # add variant LD pairs to result dictionary
            results[mut2var_dict.get(focal_mutation)] = indexes

    # set up threads for parallel processing
    threads = [
    threading.Thread(target=thread_worker, args=(j,)) for j in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # return LD dictionary
    return results

def _gwas(genos_case,genos_control):
    """
    Perform chi-squared to get variant p-values
    and odds ratios

    Parameters
    ----------
    genos_case : np.array
        genotypes for cases
    genos_control : np.array
        genotypes for controls

    Returns
    -------
    float
		variant odds ratio
    float
    	variant p-value
    """
    # number of cases with each allele count
    case_AA = np.sum(genos_case>1,axis=1)  # two alt alleles
    case_AB = np.sum(genos_case==1,axis=1) # one ref, one alt
    case_BB = np.sum(genos_case==0,axis=1) # two ref alleles
    # number of controls with each allele count
    control_AA = np.sum(genos_control>1,axis=1) # two alt alleles
    control_AB = np.sum(genos_control==1,axis=1) # one ref, one alt
    control_BB = np.sum(genos_control==0,axis=1) # two ref alleles
    
    R = case_BB + case_AA + case_AB # total case alleles
    S = control_BB + control_AA + control_AB # total control alleels
    n0 = case_AA+control_AA # number alt homozygous
    n1 = case_AB+control_AB # number alt het
    n2 = case_BB+control_BB # number ref homozygous
    N = R+S # total alleles cases + controls
    # compute expected counts
    exp_counts = np.array([[(2*R*(2*n0+n1))/(2*N),(2*R*(n1+2*n2))/(2*N)],
                           [(2*S*(2*n0+n1))/(2*N),(2*S*(n1+2*n2))/(2*N)]])
    # compute observed counts
    obs_counts = np.array([[2*case_AA+case_AB, case_AB+2*case_BB],
                           [2*control_AA+control_AB, control_AB+2*control_BB]])
    # compute p-value with chi-squared
    chistat,pval = stats.chisquare(obs_counts.ravel(),f_exp=exp_counts.ravel(),ddof=1)
    # create 2x2 contingency table
    case_A, case_B = 2*case_AA+case_AB, case_AB+2*case_BB
    control_A, control_B = 2*control_AA+control_AB, control_AB+2*control_BB
    # calculate odds ratio (OR), if case ref or control alt 0 
    # return 1 as OR, which will equate to 0 when log(OR)
    try:
        OR = (case_A*control_B)/(case_B*control_A)
        return OR, pval
    except ZeroDivisionError:
        return 1, pval

def _decrease_training_samples(m,h2,pop,num,prefix):
    """
    Decrease the number of training cases and controls
    to be used for a GWAS

    Parameters
    ----------
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    num : int
        Number of African cases to use for GWAS,
        an equal number of controls are used
    prefix : str
        Output file path

    Returns
    -------
    np.array
    	randomly selected cases
    np.array
    	randomly selected controls
    """
    # open true PRS file
    f = h5py.File(prefix+'true_prs/prs_m_{}_h2_{}.hdf5'.format(1000,0.5), 'r')
    # extract cases and controls
    cases,controls = f["train_cases_{}".format(pop)][()], f["train_controls_{}".format(pop)][()]
    # randomly select num cases and controls
    sub_cases = np.random.choice(cases,size=num,replace=False)
    sub_controls = np.random.choice(controls,size=num,replace=False)
    f.close()
    # return new subsetted cases/controls
    return sub_cases,sub_controls

def _compute_maf(tree,prefix,pop):
    """
    Calculate minor allele frequencies (MAFs)
    for a given simulation tree

    Parameters
    ----------
    tree : msprime.TreeSequence
        simulation tree for a given population
    prefix : str
        Output file path
    pop : str
        population to compute MAFs for

    Returns
    -------
    np.array
    	MAFs for all simulated genotypes
    """
    # check if output exists
    if not os.path.isfile(prefix+"emp_prs/maf_{}.txt".format(pop)):
    	# track progress
        pbar = tqdm.tqdm(total=tree.num_sites)
        # initialize array
        maf = np.zeros(shape=tree.num_sites,dtype=float)
        # loop through all simulated variants 
        for ind,var in enumerate(tree.variants()):
        	# convert phased haplotypes to genotype
            genos_diploid = return_diploid_genos(var.genotypes,tree)
            # calculate allele frequency of ALT allele
            freq = np.sum(genos_diploid,axis=1)/(2*genos_diploid.shape[1])
            # convert to MAF if allele frequency > 0.5
            if freq < 0.5: maf[ind] = freq
            else: maf[ind] = 1-freq
            # update progress bar 
            pbar.update(1)
        # save results and output
        np.savetxt(prefix+"emp_prs/maf_{}.txt".format(pop),maf)
        return maf
    # if output exists, load and return file
    else: return np.loadtxt(prefix+"emp_prs/maf_{}.txt".format(pop))

def _load_data(weight,selection,path_tree_CEU,path_tree_YRI,prefix,m,h2,num2decrease):
    """
    Load simulation trees, summary statistics, training and testing
    samples, and sample IDs

    Parameters
    ----------
    selection: str
        Population to use for SNP selection
    weight: str
        Population to use for SNP weights
    path_tree_CEU : str
        Path to simulated tree containing CEU individuals
    path_tree_YRI : str
        Path to simulated tree containing YRI individuals
    prefix : str
        Output file path
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    num2decrease: int, optional
        Number of non-European samples to use in GWAS

    Returns
    -------
    dict
    	simulated population sequences
    	key - population
    	value - msprime.TreeSequence
    dict
    	summary statistics in each population
    	key - population
    	value - pd.DataFrame
    dict
    	training cases for each population
    	key - population
    	value - np.array
    dict
    	training controls for each population
    	key - population
    	value - np.array
    np.array
    	sample IDs
    """
    # populations to load for each snp selection and weighting strategy
    pop_dict = {"ceu":["ceu"],"yri":["yri"],"meta":["ceu","yri"],"la":["ceu","yri"]}
    # get the set of populations needed across strategies
    pops2load = set(pop_dict.get(weight)+pop_dict.get(selection))
    # load simulation trees
    trees = {"ceu":msprime.load(prefix+path_tree_CEU),"yri":msprime.load(prefix+path_tree_YRI),
            "meta":msprime.load(prefix+"trees/tree_all.hdf")}
    # check if European and African sample size is the same
    if num2decrease == None: # African sample size equivalenet to Europeans
    	# load true prs file
        f = h5py.File(f'{prefix}true_prs/prs_m_{m}_h2_{h2}.hdf5', 'r')
        # get indexes of cases
        # 200000 subtracted from YRI indices so that they refer to indices in the African-only tree 
        # need to update to avoid hard-coding
        train_cases = {"ceu":f["train_cases_ceu"][()],"yri":f["train_cases_yri"][()]-200000,
                       "meta":np.append(f["train_cases_ceu"][()],f["train_cases_yri"][()]),
                       "la":f["train_cases_yri"][()]} # only needed to get number of samples used
        # get indexes of controls
        # 200000 subtracted from YRI indices so that they refer to indices in the African-only tree 
        # need to update to avoid hard-coding
        # Not subtracted from meta or la because this is just used to get number of samples used (not for subsetting)
        train_controls = {"ceu":f["train_controls_ceu"][()],"yri":f["train_controls_yri"][()]-200000,
                       "meta":np.append(f["train_controls_ceu"][()],f["train_controls_yri"][()]),
                       "la":f["train_controls_yri"][()]}
        labels_all = f["labels"][()]
        f.close()
    else: # African sample size is smaller than in Europeans
        sub_yri_case,sub_yri_control = _decrease_training_samples(m,h2,"yri",num2decrease,prefix)
        f = h5py.File(f'{prefix}true_prs/prs_m_{m}_h2_{h2}.hdf5', 'r')
        # get indexes of cases
        # 200000 subtracted from YRI indices so that they refer to indices in the African-only tree 
        # need to update to avoid hard-coding
        # Not subtracted from meta or la because this is just used to get number of samples used (not for subsetting)
        train_cases = {"ceu":f["train_cases_ceu"][()],"yri":sub_yri_case-200000,
                       "meta":np.append(f["train_cases_ceu"][()],sub_yri_case),
                       "la":sub_yri_case} 
        # get indexes of controls
        # 200000 subtracted from YRI indices so that they refer to indices in the African-only tree 
        # need to update to avoid hard-coding
        # Not subtracted from meta or la because this is just used to get number of samples used (not for subsetting)
        train_controls = {"ceu":f["train_controls_ceu"][()],"yri":sub_yri_control-200000,
                          "meta":np.append(f["train_controls_ceu"][()],sub_yri_control),
                          "la":sub_yri_control}
        # get sample labels
        labels_all = f["labels"][()]
        f.close() 
    # initialize object with summary statistics
    sumstats = {"ceu":None,"yri":None}
    # for each population needed get summary statistics
    for pop in pops2load:
    	# compute or load summary statistics for a population
        sumstats[pop] = _compute_summary_stats(m,h2,trees[pop],
                                               train_cases[pop],
                                               train_controls[pop],
                                               pop,prefix)
    # if weighting or snp selection requires meta, perform meta
    if weight == "meta" or weight == "la" or selection == "meta":
        sumstats["meta"] = _perform_meta(train_cases,m,h2,prefix)
    # return simulation trees, summary statistics, training and testing, and labels
    return trees,sumstats,train_cases,train_controls,labels_all

def _write_output(prs,labels,prefix,m,h2,r2,p,selection,weight,
                    num_cases_weight,num_cases_selection):
    """
    Write GWAS estimated PRS to hdf file

    Parameters
    ----------
   	prs : np.array
       GWAS estimated PRS for each individual
    labels : np.array
       Labels of individuals with PRS
    prefix : str
        Output file path
    m : int
        Number of causal variants
    h2 : float
        Heritability due to genetics
    r2 : float
        LD r2 to use for summary statistic clumping
    p : float
        P-value to use for summary statistic thresholding
    selection: str
        Population to use for SNP selection
    weight: str
        Population to use for SNP weights
    num_cases_weight : int
    	Number of cases used to calculate PRS weights
    num_cases_selection : int
    	Number of cases used to select PRS SNPs
    """
    # get total number of samples
    n_all = len(labels)
    # set output file name
    outfile = f'emp_prs/emp_prs_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{selection}'+\
    		  f'_snps_{num_cases_selection}cases_{weight}_weights_'+\
    		  f'{num_cases_weight}cases.hdf5'
    # save output as hdf
    with h5py.File(prefix+outfile, 'w') as f:
        f.create_dataset("labels",(n_all,),data=labels)
        f.create_dataset("X",(n_all,),dtype=float,data=prs)
    return
