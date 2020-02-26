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

from .true_risk import return_diploid_genos
from .true_risk import calc_prs_tree, calc_prs_vcf

def create_emp_prs(m,h2,n_admix,prefix,p=0.01,r2=0.2,
    vcf_file = "admixed_data/output/admix_afr_amer.query.vcf",
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
    trees,sumstats,train_cases,train_controls = _load_data(snp_weighting,snp_selection,path_tree_CEU,
                                                                  path_tree_YRI,prefix,m,h2,
                                                                  num2decrease)
    snps = _select_variants(sumstats[snp_selection],trees[snp_selection],m,h2,
                           p,r2,snp_selection,prefix,ld_distance,num_threads)

    # if snp_weighting != "la":
    #     weights = np.log(sumstats[snp_weighting].reindex(snps,fill_value=1)["OR"])
    #     prs_ceu = calc_prs_tree(dict(zip(snps,weights)),trees[0])
    #     prs_yri = calc_prs_tree(dict(zip(snps,weights)),trees[1])
    #     prs_admix = calc_prs_vcf(prefix+vcf_file,dict(zip(snps,weights)),n_admix)
    #     prs_all = np.concatenate((prs_ceu,prs_yri,prs_admix),axis=None)

    # else:
    #     return
    return
    # Check weighting and snp selection are proper values or exit program
    # print(f"\nConstruction of empirical PRS with {snp_selection.upper()} selected snps and {snp_weighting.upper()} based weighting\n\n")
    # trees,sumstats,train_cases,train_controls,labels = _load_data(snp_weighting,path_tree_CEU,path_tree_YRI,prefix)
    # pops = ["ceu","yri"]
    # # Need to allow for different population to be used for snp_selection... should probably just compute both sum stats even if they aren't used right away
    # # Also need to allow samples to be smaller for summary statistics
    # if snp_weighting in pops:
    #     # sumstats = _compute_summary_stats(m,h2,trees[pops.index(snp_weighting)],
    #     #                                 train_cases,train_controls,snp_weighting,prefix)
    #     snps = _select_variants(sumstats,trees[pops.index(snp_selection)],m,h2,p,r2,
    #                             snp_selection,prefix,max_distance,num_threads)
    #     weights = np.log(sumstats.loc[snps,"OR"])

    #     prs_ceu = calc_prs_tree(dict(zip(snps,weights)),trees[0])
    #     prs_yri = calc_prs_tree(dict(zip(snps,weights)),trees[1])
    #     prs_admix = calc_prs_vcf(prefix+vcf_file,dict(zip(snps,weights)),n_admix)
    #     prs_all = np.concatenate((prs_ceu,prs_yri,prs_admix),axis=None)

    # else:
    #     # sumstats_ceu = _compute_summary_stats(m,h2,trees[0],train_cases[0],train_controls[0],"ceu",prefix)
    #     # sumstats_yri = _compute_summary_stats(m,h2,trees[1],train_cases[1],train_controls[1],"yri",prefix) 
    #     # sumstats_all = [sumstats_ceu,sumstats_yri]
    #     snps = _select_variants(sum_stats_all[pops.index(snp_selection)],trees[pops.index(snp_selection)],m,h2,p,r2,
    #                             snp_selection,prefix,max_distance,num_threads)

    #     if snp_weigting == "meta":
    #         weights = _perform_meta()
    #         prs_ceu = calc_prs_tree(dict(zip(snps,weights)),trees[0])
    #         prs_yri = calc_prs_tree(dict(zip(snps,weights)),trees[1])
    #         prs_admix = calc_prs_vcf(vcf_file,dict(zip(snps,weights)),n_admix)
    #         prs_all = np.concatenate((prs_ceu,prs_yri,prs_admix),axis=None)

    #     else:
    #         prs_ceu = None
    #         prs_yri = None
    #         prs_admix = None
    #         prs_all = np.concatenate((prs_ceu,prs_yri,prs_admix),axis=None)

    # _write_output(prs_all,labels,prefix,m,h2,r2,p,snp_selection,snp_weighting,num_cases_weight,num_cases_selection)
    # return

# def calc_prs_vcf_LA(): 
#     prs = np.zeros(n_admix)
#     with open(vcf_file) as f:
#         ind=0
#         for line in f:
#             if line[:6]== "#CHROM":
#                 sample_ids_admix = line.split("\n")[0].split("\t")[9:]
#             if line[0] != "#":
#                 if ind in var_dict.keys():
#                     data = line.split("\n")[0].split("\t")[9:]
#                     genotype = np.array([np.array(hap.split("|")).astype(int).sum() for hap in data])
#                     prs=prs+(genotype*var_dict[ind])
#                 ind+=1
#     return prs

def _load_data(weight,selection,path_tree_CEU,path_tree_YRI,prefix,m,h2,num2decrease):
    pop_dict = {"ceu":["ceu"],"yri":["yri"],"meta":["ceu","yri"],"la":["ceu","yri"]}
    pops2load = set(pop_dict.get(weight)+pop_dict.get(selection))

    trees = {"ceu":msprime.load(prefix+path_tree_CEU),"yri":msprime.load(prefix+path_tree_YRI)}
    
    if num2decrease == None:
        f = h5py.File(prefix+'true_prs/prs_m_{}_h2_{}.hdf5'.format(1000,0.5), 'r')
        train_cases = {"ceu":f["train_cases_ceu"][()],"yri":f["train_cases_yri"][()]-200000}
        train_controls = {"ceu":f["train_controls_ceu"][()],"yri":f["train_controls_yri"][()]-200000}
        f.close()
    else:
        sub_yri_case,sub_yri_control = _decrease_training_samples(m,h2,"yri",num2decrease,prefix)
        f = h5py.File(prefix+'true_prs/prs_m_{}_h2_{}.hdf5'.format(1000,0.5), 'r')
        train_cases = {"ceu":f["train_cases_ceu"][()],"yri":sub_yri_case-200000}
        train_controls = {"ceu":f["train_controls_ceu"][()],"yri":sub_yri_control-200000}
        f.close() 
    
    sumstats = {"ceu":None,"yri":None}
    for pop in pops2load:
        sumstats[pop] = _compute_summary_stats(m,h2,trees[pop],
                                               train_cases[pop],
                                               train_controls[pop],
                                               pop,prefix)
    if weight == "meta":
        sumstats["meta"] = _perform_meta(train_cases,m,h2,prefix)
    return trees,sumstats,train_cases,train_controls

# def _write_output(prs,labels,prefix,m,h2,r2,p,selection,weight,num_cases_weight,num_cases_selection):
#     with h5py.File(prefix+f'emp_prs/prs_m_{m}_h2_{h2}_r2_{r2}_p_{p}_{selection}_'\
#                     +f'snps_{}cases_{weight}_weights_{}cases.hdf5', 'w') as f:
#         f.create_dataset("labels",(n_all,),data=labels)
#         f.create_dataset("X",(n_all,),dtype=float,data=prs)
#     return

def _perform_meta(train_cases,m,h2,prefix):
    if not os.path.isfile(prefix+f"emp_prs/meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases[0])}"+\
                                 f"_casesYRI_{len(train_cases[1])}.txt"):
        print("\nPerforming a fixed_effects meta between CEU and YRI summary statistics")
        os.system("Rscript simulation/compute_meta_sum_stats.R " +\
                 f"{prefix}emp_prs/gwas_m_{m}_h2_{h2}_pop_ceu_cases_{len(train_cases[0])}.txt " +\
                 f"{prefix}emp_prs/gwas_m_{m}_h2_{h2}_pop_yri_cases_{len(train_cases[1])}.txt " +\
                 f"{prefix}emp_prs/meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases[0])}_casesYRI_{len(train_cases[1])}.txt")
    
    return pd.read_csv(prefix+f"emp_prs/meta_m_{m}_h2_{h2}_casesCEU_{len(train_cases[0])}"+\
                                 f"_casesYRI_{len(train_cases[1])}.txt",sep="\t",index_col=0)

def _select_variants(sum_stats,tree,m,h2,p,r2,pop,prefix,max_distance,num_threads):
    print("-----------------------------------")
    print("Selecting variants for PRS building")
    print("-----------------------------------")
    print(f"Population used for LD clumping = {pop}")
    print(f"Parameters: p-value = {p} and r2 = {r2}")
    prs_vars = sum_stats[sum_stats["p-value"] < p].sort_values(by=["p-value"]).index
    print(f"# variants with p < {p}: {len(prs_vars)}")
    clumped_prs_vars = _ld_clump(tree,prs_vars,m,h2,pop,r2,p,
                                 prefix,max_distance,num_threads)
    print(f"# variants after clumping: {len(clumped_prs_vars)}")
    print("-----------------------------------")
    return clumped_prs_vars

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
        sum_stats.to_csv(prefix+"emp_prs/gwas_m_{}_h2_{}_pop_{}_cases_{}.txt".format(m,h2,pop,len(train_cases)),sep="\t",index=True)
        return sum_stats
    else: 
        sum_stats = pd.read_csv(prefix+"emp_prs/gwas_m_{}_h2_{}_pop_{}_cases_{}.txt".format(m,h2,pop,len(train_cases)),sep="\t",index_col=0)
        return sum_stats

def _ld_clump(tree,variants,m,h2,pop,r2,p,prefix,max_distance,num_threads):
    if not os.path.isfile(prefix+f"emp_prs/clumped_prs_vars_m_{m}_h2_{h2}_pop_{pop}_r2_{r2}_p{p}.txt"):
        print("Clumping variants...")
        var2mut,mut2var = _get_var_mut_maps(tree)
        tree_ld = tree.simplify(filter_sites=True)
        ld_struct = _compute_ld_variants(tree_ld,variants,r2,var2mut,mut2var,max_distance,num_threads)

        clumped_variants = [variants[0]]
        for v in range(1,len(variants)):
            add, i = True, 0
            while add and i < len(clumped_prs):
                if variants[v] in ld_struct.get(clumped_prs[i]):
                    add = False
                i+=1
            if add: clumped_variants.append(variants[v])
        np.savetxt(prefix+f"emp_prs/clumped_prs_vars_m_{m}_h2_{h2}_pop_{pop}_r2_{r2}_p{p}.txt")
        return clumped_variants
    else: 
        return np.loadtxt(prefix+f"emp_prs/clumped_prs_vars_m_{m}_h2_{h2}_pop_{pop}_r2_{r2}_p{p}.txt")

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
    case_ref = 2*np.sum(genos_case==0,axis=1)+np.sum(genos_case==1,axis=1)
    case_alt = 2*np.sum(genos_case==2,axis=1)+np.sum(genos_case==1,axis=1)
    control_ref = 2*np.sum(genos_control==0,axis=1)+np.sum(genos_control==1,axis=1)
    control_alt = 2*np.sum(genos_control==2,axis=1)+np.sum(genos_case==1,axis=1)
    obs = np.array([[case_ref[0],case_alt[0]],[control_ref[0],control_alt[0]]])
    chi2, pval, dof, ex = stats.chi2_contingency(obs)

    try:
        OR = (case_alt*control_ref)/(case_ref*control_alt)
        return OR[0], pval
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