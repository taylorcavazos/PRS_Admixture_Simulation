"""
Want to be able to change weighting for empirical PRS (meta,LA,CEU), 
change number of training samples used,
change p-value, and r2,
and finally change SNP selection (future use)
"""

import numpy as np, pandas as pd, math
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
    sim = prefix.split('sim')[1].split('/')[0]
    trees,sumstats,train_cases,train_controls,labels = _load_data(snp_weighting,snp_selection,path_tree_CEU,
                                                                  path_tree_YRI,prefix,m,h2,
                                                                  num2decrease)
    snps = _select_variants(sumstats[snp_selection],trees[snp_selection],m,h2,
                           p,r2,snp_selection,prefix,ld_distance,num_threads,num2decrease)
    causal_inds =  np.linspace(0, trees["ceu"].num_sites, m, dtype=int,endpoint=False)
    _write_allele_freq_bins(sim,causal_inds,trees,prefix,snp_selection,prefix+vcf_file,m,h2,r2,p,train_cases,causal=True)
    _write_allele_freq_bins(sim,snps.astype(int),trees,prefix,snp_selection,prefix+vcf_file,m,h2,r2,p,train_cases) 

    if os.path.isfile(f"{prefix}emp_prs/emp_prs_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{snp_selection}_"+\
                      f"snps_{len(train_cases[snp_selection])}cases_{snp_weighting}_weights_"+\
                      f"{len(train_cases[snp_weighting])}cases.hdf5"):
        print(f"\nEmpirical PRS for iteration={prefix.split('sim')[1].split('/')[0]} and provided parameters exists")
        print(f"If you would like to overwrite, remove {prefix}emp_prs/emp_prs_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{snp_selection}_snps_{len(train_cases[snp_selection])}cases_{snp_weighting}_weights_{len(train_cases[snp_weighting])}cases.hdf5")
        return

    else:
        print(f"Creating empricial with {snp_selection.upper()} selected snps and {snp_weighting.upper()} weights ")
        
        if snp_weighting != "la":
            weights = np.log(sumstats[snp_weighting].reindex(snps,fill_value=1)["OR"])
            print("..... for the CEU population")
            prs_ceu = calc_prs_tree(dict(zip(snps,weights)),trees["ceu"])
            print("..... for the YRI population")
            prs_yri = calc_prs_tree(dict(zip(snps,weights)),trees["yri"])
            print("..... for the admixed population")
            prs_admix,ids_admix = calc_prs_vcf(prefix+vcf_file,dict(zip(snps,weights)),n_admix)
            prs_all = np.concatenate((prs_ceu,prs_yri,prs_admix),axis=None)
            _write_output(prs_all,labels,prefix,m,h2,r2,p,snp_selection,snp_weighting,
                    len(train_cases[snp_weighting]),len(train_cases[snp_selection]))
        else:
            _ancestry_snps_admix(snps,prefix,m,h2,r2,p,snp_selection,num2decrease)
            weights = {"ceu":np.log(sumstats["ceu"].reindex(snps,fill_value=1)["OR"]),
                       "yri":np.log(sumstats["yri"].reindex(snps,fill_value=1)["OR"]),
                       "meta":np.log(sumstats["meta"].reindex(snps,fill_value=1)["OR"])}

            print("..... for the CEU population")
            prs_ceu = calc_prs_tree(dict(zip(snps,weights["ceu"])),trees["ceu"])
            print("..... for the YRI population")
            prs_yri = calc_prs_tree(dict(zip(snps,weights["yri"])),trees["yri"])
            print("..... for the admixed population")
            
            prs_admix = calc_prs_vcf_la(prefix+vcf_file,weights,snps,n_admix,m,h2,r2,p,snp_selection,prefix,trees["ceu"].num_sites,num2decrease)
            prs_all = np.concatenate((prs_ceu,prs_yri,prs_admix),axis=None)
            _write_output(prs_all,labels,prefix,m,h2,r2,p,snp_selection,snp_weighting,
                    len(train_cases[snp_weighting]),len(train_cases[snp_selection]))
            return
    return

def calc_prs_vcf_la(vcf_file,weights,snps,n_admix,m,h2,r2,p,pop,prefix,num_sites,num2decrease):
    if num2decrease==None: anc_file = f"{prefix}admixed_data/output/admix_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{pop}_snps.result.PRS"
    else: anc_file = f"{prefix}admixed_data/output/admix_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{pop}_snps_{num2decrease}.result.PRS"
    anc = pd.read_csv(anc_file,sep="\t",index_col=0)
    sample_ids = pd.read_csv(f"{prefix}admixed_data/output/admix_afr_amer.prop.anc",
        sep="\t",index_col=0).index
    prs = np.zeros(n_admix)
    pbar = tqdm.tqdm(total=num_sites)
    with gzip.open(vcf_file,"rt") as f:
        ind=0
        for line in f:
            if line[0] != "#":
                if ind in snps:
                    data = line.split("\t")[9:]
                    genotype = np.array([np.array(hap.split("|")).astype(int).sum() for hap in data])
                    var_weighted=[]
                    for g in range(0,len(genotype)):
                        if anc.loc[ind,[sample_ids[g]+".0",sample_ids[g]+".1"]].sum() > 3:
                            var_weighted.append(genotype[g]*weights["yri"][ind])
                        elif anc.loc[ind,[sample_ids[g]+".0",sample_ids[g]+".1"]].sum() == 3:
                            var_weighted.append(genotype[g]*weights["meta"][ind])
                        elif anc.loc[ind,[sample_ids[g]+".0",sample_ids[g]+".1"]].sum() < 3:
                            var_weighted.append(genotype[g]*weights["ceu"][ind])
                    prs=prs+np.array(var_weighted)
                ind+=1
                pbar.update(1)
    return prs

def _ancestry_snps_admix(snps,prefix,m,h2,r2,p,pop,num2decrease):
    if num2decrease==None: outfile = f"{prefix}admixed_data/output/admix_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{pop}_snps.result.PRS"
    else: outfile = f"{prefix}admixed_data/output/admix_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{pop}_snps_{num2decrease}.result.PRS"
    if not os.path.isfile(outfile):
        with gzip.open(f"{prefix}admixed_data/output/admix_afr_amer.result.gz","rt") as anc:
            print("Extracting proportion ancestry at PRS variants")
            for ind,line in enumerate(anc):
                if ind == 0:
                    anc_prs = pd.DataFrame(columns=line.split("\n")[0].split("\t")[2:])

                elif ind-1 in snps:
                    anc_prs.loc[ind-1,:] = line.split("\n")[0].split("\t")[2:]

        anc_prs.to_csv(outfile,sep="\t")
    return

def _perform_meta(train_cases,m,h2,prefix):
    if not os.path.isfile(prefix+f"emp_prs/meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases['ceu'])}"+ \
                                 f"_casesYRI_{len(train_cases['yri'])}.txt"):
        print("\nPerforming a fixed_effects meta between CEU and YRI summary statistics")
        os.system("Rscript simulation/compute_meta_sum_stats.R " + \
                 f"{prefix}emp_prs/gwas_m_{m}_h2_{h2}_pop_ceu_cases_{len(train_cases['ceu'])}.txt " + \
                 f"{prefix}emp_prs/gwas_m_{m}_h2_{h2}_pop_yri_cases_{len(train_cases['yri'])}.txt " + \
                 f"{prefix}emp_prs/meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases['ceu'])}_casesYRI_{len(train_cases['yri'])}.txt")
        sum_stats = pd.read_csv(prefix+f"emp_prs/meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases['ceu'])}"+ \
                                 f"_casesYRI_{len(train_cases['yri'])}.txt",sep="\t",index_col=0)
        _plot_qq(sum_stats,prefix,f"meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases['ceu'])}_casesYRI_{len(train_cases['yri'])}")

    return pd.read_csv(prefix+f"emp_prs/meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases['ceu'])}_casesYRI_{len(train_cases['yri'])}.txt",sep="\t",index_col=0)

def _select_variants(sum_stats,tree,m,h2,p,r2,pop,prefix,max_distance,num_threads,num2decrease):
    print("-----------------------------------")
    print("Selecting variants for PRS building")
    print("-----------------------------------")
    print(f"Population used for LD clumping = {pop}")
    print(f"Parameters: p-value = {p} and r2 = {r2}")
    prs_vars = sum_stats[sum_stats["p-value"] < p].sort_values(by=["p-value"]).index
    print(f"# variants with p < {p}: {len(prs_vars)}")
    clumped_prs_vars = _ld_clump(tree,prs_vars,m,h2,pop,r2,p,
                                 prefix,max_distance,num_threads,num2decrease)
    print(f"# variants after clumping: {len(clumped_prs_vars)}")
    print("-----------------------------------")
    return clumped_prs_vars

def _compute_maf_vcf(vcf_file,var_list):
    mafs = []
    with gzip.open(vcf_file,"rt") as f:
        ind = 0
        for line in f:
            if line[0] != "#":
                if ind in var_list:
                    data = line.split("\n")[0].split("\t")[9:]
                    genotype = np.array([np.array(hap.split("|")).astype(int).sum() for hap in data])
                    genotype = genotype.reshape(1,len(genotype))
                    freq = np.sum(genotype,axis=1)/(2*genotype.shape[1])
                    if freq < 0.5: maf = freq
                    else: maf = 1-freq
                    mafs.append(maf)
                ind+=1

    return np.array(mafs)

def _return_maf_group(mafs,n_sites):
    G1 = len(mafs[mafs<0.01])/n_sites
    G2 = len(mafs[(mafs>=0.01)&(mafs<0.1)])/n_sites
    G3 = len(mafs[(mafs>=0.1)&(mafs<0.2)])/n_sites
    G4 = len(mafs[(mafs>=0.2)&(mafs<0.3)])/n_sites
    G5 = len(mafs[(mafs>=0.3)&(mafs<0.4)])/n_sites
    G6 = len(mafs[(mafs>=0.4)&(mafs<0.5)])/n_sites
    return [G1,G2,G3,G4,G5,G6]

def _write_allele_freq_bins(sim,var_list,trees,prefix,snp_selection,vcf_file,
    m,h2,r2,p,train_cases,causal=False):
    if not os.path.isfile(f"{prefix}summary/emp_maf_bins_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{snp_selection}_snps_{len(train_cases)}cases_{sim}.txt"):
        maf_ceu = _compute_maf(trees["ceu"],prefix,"ceu")[var_list]
        maf_yri = _compute_maf(trees["yri"],prefix,"yri")[var_list]
        maf_admix = _compute_maf_vcf(vcf_file,var_list)

        df = pd.DataFrame(columns=["sim","pop","0 - 0.01","0.01 - 0.1","0.1 - 0.2","0.2 - 0.3","0.3 - 0.4","0.4 - 0.5"])
        for pop,mafs in zip(["ceu","yri","admix"],[maf_ceu,maf_yri,maf_admix]):
            groups = _return_maf_group(mafs,len(mafs))
            sub_df = pd.DataFrame([[sim,pop]+groups],columns=["sim","pop","0 - 0.01","0.01 - 0.1","0.1 - 0.2","0.2 - 0.3","0.3 - 0.4","0.4 - 0.5"])
            df = df.append(sub_df, ignore_index=True)

        if not causal:
            df.to_csv(f"{prefix}summary/emp_maf_bins_m_{m}_h2_{h2}_r2_{r2}_p_{p}"+\
                        f"_{snp_selection}_snps_{len(train_cases[snp_selection])}cases_"+\
                        f"{sim}.txt",sep="\t")
        else:
            df.to_csv(f"{prefix}summary/causal_maf_bins_m_{m}_h2_{h2}_{sim}.txt",sep="\t")
    return

def _compute_summary_stats(m,h2,tree,train_cases,train_controls,pop,prefix):
    if not os.path.isfile(prefix+"emp_prs/gwas_m_{}_h2_{}_pop_{}_cases_{}.txt".format(m,h2,pop,len(train_cases))):
        print(f"Computing SNP MAF for {pop.upper()}. GWAS will be performed for SNPs with maf > 1%")
        mafs = _compute_maf(tree,prefix,pop)
        var_ids = np.where(np.array(mafs) >= 0.01)[0]

        print("Running GWAS for population = {}".format(pop.upper()))
        print("------------------- # cases = {}".format(len(train_cases)))
        print("------------------- # sites = {}".format(len(var_ids)))
        print("\n")

        sum_stats_arr = np.empty(shape=(len(var_ids),3))
        pbar = tqdm.tqdm(total=tree.num_sites)

        var_loc = 0
        for var in tree.variants():
            if var.site.id in var_ids:
                genos = return_diploid_genos(var.genotypes,tree)
                genos_cases = genos[:,train_cases]
                genos_controls = genos[:,train_controls]
                OR,pval = _gwas(genos_cases,genos_controls)
                sum_stats_arr[var_loc]=[var.site.id,OR,pval]
                var_loc+=1
            pbar.update(1)
        sum_stats = pd.DataFrame(sum_stats_arr,columns=["var_id","OR","p-value"])
        sum_stats = sum_stats.replace([np.inf, -np.inf], np.nan)
        sum_stats.dropna(inplace=True)
        sum_stats = sum_stats.set_index("var_id").sort_index()
        sum_stats.to_csv(f"{prefix}emp_prs/gwas_m_{m}_h2_{h2}_pop_{pop}_cases_{len(train_cases)}.txt",sep="\t",index=True)
        _plot_qq(sum_stats,prefix,f"gwas_m_{m}_h2_{h2}_pop_{pop}_cases_{len(train_cases)}")
        return sum_stats
    else: 
        sum_stats = pd.read_csv(f"{prefix}emp_prs/gwas_m_{m}_h2_{h2}_pop_{pop}_cases_{len(train_cases)}.txt",sep="\t",index_col=0)
        return sum_stats

def _plot_qq(sum_stats,prefix,outfile):
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
    plt.ylim(0,300)
    sns.despine()
    plt.savefig(f"{prefix}summary/{outfile}_QQ.png",type="png",bbox_inches="tight",dpi=400)



def _ld_clump(tree,variants,m,h2,pop,r2,p,prefix,max_distance,num_threads,num2decrease):
    if num2decrease == None: path = prefix+f"emp_prs/clumped_prs_vars_m_{m}_h2_{h2}_pop_{pop}_r2_{r2}_p{p}.txt"
    else: path = prefix+f"emp_prs/clumped_prs_vars_m_{m}_h2_{h2}_pop_{pop}_r2_{r2}_p{p}_cases_{num2decrease}.txt"
    if not os.path.isfile(path):
        print("Clumping variants...")
        var2mut,mut2var = _get_var_mut_maps(tree)
        tree_ld = tree.simplify(filter_sites=True)
        ld_struct = _compute_ld_variants(tree_ld,variants,r2,var2mut,mut2var,max_distance,num_threads)

        clumped_variants = [variants[0]]
        for v in range(1,len(variants)):
            add, i = True, 0
            while add and i < len(clumped_variants):
                if variants[v] in ld_struct.get(clumped_variants[i]):
                    add = False
                i+=1
            if add: clumped_variants.append(variants[v])
        np.savetxt(path,clumped_variants)
        return np.array(clumped_variants)
    else: 
        return np.loadtxt(path)

def _get_var_mut_maps(tree):
    var2mut, mut2var = {}, {}
    for mut in tree.mutations():
        mut2var[mut.id]=mut.site
        var2mut[mut.site]=mut.id
    return var2mut, mut2var

def _compute_ld_variants(tree,focal_vars,r2,var2mut_dict,
            mut2var_dict,max_distance,num_threads):
    results = {}
    num_threads = min(num_threads, len(focal_vars))

    def thread_worker(thread_index):
        ld_calc = msprime.LdCalculator(tree)
        chunk_size = int(math.ceil(len(focal_vars) / num_threads))
        start = thread_index * chunk_size
        for focal_var in focal_vars[start: start + chunk_size]:
            focal_mutation = var2mut_dict.get(focal_var)
            np.seterr(under='ignore')
            a = ld_calc.get_r2_array(
                focal_mutation, max_distance=max_distance,
                direction=msprime.REVERSE)
            a[np.isnan(a)] = 0
            rev_indexes = focal_mutation - np.nonzero(a >= r2)[0] - 1
            a = ld_calc.get_r2_array(
                focal_mutation, max_distance=max_distance,
                direction=msprime.FORWARD)
            a[np.isnan(a)] = 0
            fwd_indexes = focal_mutation + np.nonzero(a >= r2)[0] + 1
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

def _gwas(genos_case,genos_control):
    """
    Run chi-squared to extract p-values and effect sizes
    """
    case_AA = np.sum(genos_case>1,axis=1) 
    case_AB = np.sum(genos_case==1,axis=1)
    case_BB = np.sum(genos_case==0,axis=1)
    control_AA = np.sum(genos_control>1,axis=1)
    control_AB = np.sum(genos_control==1,axis=1)
    control_BB = np.sum(genos_control==0,axis=1)

    R = case_BB + case_AA + case_AB
    S = control_BB + control_AA + control_AB
    n0 = case_AA+control_AA
    n1 = case_AB+control_AB
    n2 = case_BB+control_BB
    N = R+S

    exp_counts = np.array([[(2*R*(2*n0+n1))/(2*N),(2*R*(n1+2*n2))/(2*N)],
                           [(2*S*(2*n0+n1))/(2*N),(2*S*(n1+2*n2))/(2*N)]])
    obs_counts = np.array([[2*case_AA+case_AB, case_AB+2*case_BB],
                           [2*control_AA+control_AB, control_AB+2*control_BB]])
    chistat,pval = stats.chisquare(obs_counts.ravel(),f_exp=exp_counts.ravel(),ddof=1)

    case_A, case_B = 2*case_AA+case_AB, case_AB+2*case_BB
    control_A, control_B = 2*control_AA+control_AB, control_AB+2*control_BB
    
    try:
        OR = (case_A*control_B)/(case_B*control_A)
        return OR, pval
    except ZeroDivisionError:
        return 1, pval

def _decrease_training_samples(m,h2,pop,num,prefix):
    f = h5py.File(prefix+'true_prs/prs_m_{}_h2_{}.hdf5'.format(1000,0.5), 'r')
    cases,controls = f["train_cases_{}".format(pop)][()], f["train_controls_{}".format(pop)][()]
    sub_cases = np.random.choice(cases,size=num,replace=False)
    sub_controls = np.random.choice(controls,size=num,replace=False)
    f.close()
    return sub_cases,sub_controls

def _compute_maf(tree,prefix,pop):
    if not os.path.isfile(prefix+"emp_prs/maf_{}.txt".format(pop)):
        pbar = tqdm.tqdm(total=tree.num_sites)
        maf = np.zeros(shape=tree.num_sites,dtype=float)
        for ind,var in enumerate(tree.variants()):
            genos_diploid = return_diploid_genos(var.genotypes,tree)
            freq = np.sum(genos_diploid,axis=1)/(2*genos_diploid.shape[1])
            if freq < 0.5: maf[ind] = freq
            else: maf[ind] = 1-freq 
            pbar.update(1)
        np.savetxt(prefix+"emp_prs/maf_{}.txt".format(pop),maf)

        return maf
    else: return np.loadtxt(prefix+"emp_prs/maf_{}.txt".format(pop))

def _load_data(weight,selection,path_tree_CEU,path_tree_YRI,prefix,m,h2,num2decrease):
    pop_dict = {"ceu":["ceu"],"yri":["yri"],"meta":["ceu","yri"],"la":["ceu","yri"]}
    pops2load = set(pop_dict.get(weight)+pop_dict.get(selection))

    trees = {"ceu":msprime.load(prefix+path_tree_CEU),"yri":msprime.load(prefix+path_tree_YRI),
            "meta":msprime.load(prefix+"trees/tree_all.hdf")}
    
    if num2decrease == None:
        f = h5py.File(f'{prefix}true_prs/prs_m_{m}_h2_{h2}.hdf5', 'r')
        train_cases = {"ceu":f["train_cases_ceu"][()],"yri":f["train_cases_yri"][()]-200000,
                       "meta":np.append(f["train_cases_ceu"][()],f["train_cases_yri"][()]),
                       "la":f["train_cases_yri"][()]}
        train_controls = {"ceu":f["train_controls_ceu"][()],"yri":f["train_controls_yri"][()]-200000,
                       "meta":np.append(f["train_controls_ceu"][()],f["train_controls_yri"][()]),
                       "la":f["train_controls_yri"][()]}
        labels_all = f["labels"][()]
        f.close()
    else:
        sub_yri_case,sub_yri_control = _decrease_training_samples(m,h2,"yri",num2decrease,prefix)
        f = h5py.File(f'{prefix}true_prs/prs_m_{m}_h2_{h2}.hdf5', 'r')
        train_cases = {"ceu":f["train_cases_ceu"][()],"yri":sub_yri_case-200000,
                       "meta":np.append(f["train_cases_ceu"][()],sub_yri_case),
                       "la":sub_yri_case}
        train_controls = {"ceu":f["train_controls_ceu"][()],"yri":sub_yri_control-200000,
                          "meta":np.append(f["train_controls_ceu"][()],sub_yri_control),
                          "la":sub_yri_control}
        labels_all = f["labels"][()]
        f.close() 
    
    sumstats = {"ceu":None,"yri":None}
    for pop in pops2load:
        sumstats[pop] = _compute_summary_stats(m,h2,trees[pop],
                                               train_cases[pop],
                                               train_controls[pop],
                                               pop,prefix)
    if weight == "meta" or weight == "la" or selection == "meta" or selection == "la":
        sumstats["meta"] = _perform_meta(train_cases,m,h2,prefix)
    return trees,sumstats,train_cases,train_controls,labels_all

def _write_output(prs,labels,prefix,m,h2,r2,p,selection,weight,
                    num_cases_weight,num_cases_selection):
    n_all = len(labels)
    with h5py.File(prefix+f'emp_prs/emp_prs_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{selection}_snps_{num_cases_selection}cases_{weight}_weights_{num_cases_weight}cases.hdf5', 'w') as f:
        f.create_dataset("labels",(n_all,),data=labels)
        f.create_dataset("X",(n_all,),dtype=float,data=prs)
    return
