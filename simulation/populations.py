import msprime
import math
import pandas as pd
import numpy as np
import os
import gzip

def simulate_populations(N_CEU, N_YRI, N_MATE, N_ADMIX, rmap_file, N_CHB=0,
						prefix="output/sim1/",chrom="20"):
	"""
	Function to simulate European and African populations based on
	msprime's out_of_africa model and Admixed (European + African). 
	Outputs vcf of sample genotypes and sample map
	for input into admixture simulations.

	Parameters
	----------
	N_CEU : int
		Number of samples of European ancestry
	N_YRI : int
		Number of samples of African ancestry
	N_MATE : int
		Number of samples to use for admixture simulation
	N_ADMIX : int
		Number of admixed samples to generate
	rmap_file : msprime.simulations.RecombinationMap
		Recombination map for a reference chromosome
	N_CHB : int, optional
		Number of samples of Asian ancestry
	prefix : str, optional
		Output file path
	chrom : str, optional
		Chromosome corresponding to rmap_file

	"""
	# Simulate Europeans and Africans
	_simulate_out_of_afr(N_CEU,N_YRI,N_CHB,N_MATE,rmap_file,prefix,chrom)
	print("Simulating admixed populations")
	# Simulate admixed individuals
	_simulate_admix(N_ADMIX,prefix)

	return None

def _simulate_admix(N_ADMIX,prefix):
	"""
	The simulation software used to create admixed individuals two reference populations
	was extracted from the rfmix [github repo](https://watermark.silverchair.com/kwy228.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAnIwggJuBgkqhkiG9w0BBwagggJfMIICWwIBADCCAlQGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMWrEIYqOMT4j4y3oRAgEQgIICJbzJZUkC_VBFfnaB4ssmvU42YSZfaWXAcQ8i0dEW6R7w-0pI_1x6MM20L9Pc1lblAP8LOpUsRF3VwIB9SZzOCN1HRt5XW3M0C5WLtHH_Fx0pVDblNY5uPbidF7xSgnNL98lzxJvecrBdsZjatOd-_xAqj84s87ksnw3ddfSY12xU7CgYEg-u17uEkBXnyacXDobbAj-Eu3m-RTpsdd3XB_r7YmGla8qk5FPNyWNQbNyo1MuUP9bDaNK7XOQCJ7veGNMRUM5K9z025V5rtr0XbIzzeskRwHyfJpe57XRymieAJhen5GTBzl9iq0sgaEtXZ7CvUHpmJSLWGeaTJQQWO39fsRpUUaJWypIas6Kioa-PYnG_TZqjqFjWBC6LLhrLdiv7Yr6YFq452RBqy-p2tEvrqmEU97gxVX1Rtdfv2ke4MN-Mf7WLaTAnIZgknNeg2iWSYgIsFOdDE62xNgf0rlqa9PGr7wYbgdE353jNDVsQ3jW-glbM5dwoQCiLkCOAsrxW7v966Ai67X6FZPXyYwehK_QOowx-9O2d3LpxywV-b5KuYCooBPEU1zlL8G5EBVuK6HvOUOKzvHGxzvLUXoDmUuF057JcO_CVbqoKGisPj6M_mQVuXwtu56q6KUkuwdMk2jwtzOknYyGbtr-FcLuh72S85VuJ9xLYCtXb7_eq4E7rtTjXXMwDvUmQDhPjCPYbOclEGsjI3K4sR9Z4tXskR1kE8w) 
	of Mark Koni Wright. The simulate function was used in Maples et. al. 2013 to test
	their rfmix software's ability to perform local ancestry inference.
	Parameters
	----------
	N_ADMIX : int
		Number of admixed samples to generate
	prefix : str
		Output file path
	"""
	os.system(f"./simulation/simulate-admixed -f {prefix}admixed_data/input/ceu_yri_genos.vcf.gz -m {prefix}admixed_data/input/ceu_yri_map.txt \
				-G 8 -c 20 -g required_data/genetic_map_GRCh37_chr20_rfmix.txt -o {prefix}admixed_data/output/admix_afr_amer \
				-s {N_ADMIX} --growth-rate=1.5")
	os.system(f"gzip {prefix}admixed_data/output/admix_afr_amer.query.vcf")
	os.system(f"gzip {prefix}admixed_data/output/admix_afr_amer.result")
	return

def _simulate_out_of_afr(N_CEU, N_YRI, N_CHB, N_MATE, rmap_file, prefix, chrom):
	"""
	Function for creating European, African, and admixed populations.
	Samples are split into mating (to be used for generating admixed individuals)
	and non-mating (to be used for PRS Training/Testing). 

	Parameters
	----------
	N_CEU : int
		Number of samples of European ancestry
	N_YRI : int
		Number of samples of African ancestry
	N_CHB : int
		Number of samples of Asian ancestry
	N_MATE : int
		Number of samples to use for admixture simulation
	rmap_file : msprime.simulations.RecombinationMap
		Recombination map for a reference chromosome
	prefix : str
		Output file path
	chrom : str
		Chromosome corresponding to rmap_file
	"""

	print("--------------------")
	print("Population Breakdown")
	print("--------------------")
	print("Number CEU: {}".format(N_CEU))
	print("Number YRI: {}".format(N_YRI))
	print("Number for mating: {}".format(N_MATE))
	print("--------------------")
	# Load chromosome recombination map
	rmap = msprime.RecombinationMap.read_hapmap(rmap_file)
	# Create population tree of all samples
	tree = _out_of_africa(N_CEU, N_YRI, N_CHB, rmap)
	# Get ancestry of samples
	sample_map = _write_sample_map(tree, N_CEU, N_YRI)
	# Save tree and write sample map
	tree.dump(prefix+"trees/tree_all.hdf")
	sample_map.to_csv(prefix+"trees/sample_map_all.txt",header=False,sep="\t",index=False)
	# Extract samples to use as founders for admixed individuals
	mate_samples = _extract_samples_for_admixture(sample_map,tree,N_MATE,prefix,chrom)
	# Get all samples
	all_data = np.array(tree.samples()).astype(np.int32)
	# Exclude samples for mating
	other_samps = [ind for ind in all_data if ind not in mate_samples]
	# Extract non-mate samples from tree and get ancestry
	tree_other = tree.simplify(samples = other_samps, filter_sites=False) # subset tree to non-mate
	sample_map_other = _write_sample_map(tree_other,N_CEU-N_MATE,N_YRI-N_MATE) # get ancestry of samples
	ceu_other_samples = sample_map_other[sample_map_other.iloc[:,1]=="CEU"] # extract Europeans
	# Get haplotypes for each diploid individual
	ceu_other_samples_hap = []
	for ind in ceu_other_samples.index:
		ceu_other_samples_hap.append(ceu_other_samples.loc[ind,2])
		ceu_other_samples_hap.append(ceu_other_samples.loc[ind,3])
	# Create tree with just Europeans
	tree_ceu_gwas = tree_other.simplify(samples=ceu_other_samples_hap,filter_sites=False)
	tree_ceu_gwas.dump(prefix+"trees/tree_CEU_GWAS_nofilt.hdf")
	# Extract Africans
	yri_other_samples = sample_map_other[sample_map_other.iloc[:,1]=="YRI"]
	# Get haplotypes for each diploid individual
	yri_other_samples_hap = []
	for ind in yri_other_samples.index:
		yri_other_samples_hap.append(yri_other_samples.loc[ind,2])
		yri_other_samples_hap.append(yri_other_samples.loc[ind,3])
	# Create tree with just Africans
	tree_yri_gwas = tree_other.simplify(samples=yri_other_samples_hap,filter_sites=False)
	tree_yri_gwas.dump(prefix+"trees/tree_YRI_GWAS_nofilt.hdf")
	return

def _extract_samples_for_admixture(sample_map,tree,N_MATE,prefix,chrom):
	"""
	Choose samples to use as founders to create admixed individuals

	Parameters
	----------
	sample_map : pd.DataFrame
		dataframe with sample ids and population code
	tree : msprime.TreeSequence
		simulated population genotypes and samples from msprime
	N_MATE : int
		Number of samples to use for admixture simulation
	prefix : str
		Output file path
	chrom : str
		Chromosome corresponding to rmap_file

	Returns
	-------
	list
		Samples used as founders for admixed samples
	"""
	# Get European sample IDs
	ceu_samples = sample_map[sample_map.iloc[:,1]=="CEU"]
	# Get African sample IDs
	yri_samples = sample_map[sample_map.iloc[:,1]=="YRI"]
	# Randomly select samples for mating from each population
	ceu_mate = ceu_samples.loc[np.random.choice(ceu_samples.index,size=N_MATE,replace=False)]
	yri_mate = yri_samples.loc[np.random.choice(yri_samples.index,size=N_MATE,replace=False)]
	# Compine results
	ALL_mate = pd.concat([ceu_mate,yri_mate])
	# Get haplotype positions for each diploid individual
	mate_samples = []
	for ind in ALL_mate.index:
		mate_samples.append(ALL_mate.loc[ind,2])
		mate_samples.append(ALL_mate.loc[ind,3])
	# Select samples from tree
	mate_samples = np.array(mate_samples).astype(np.int32)
	tree_mate = tree.simplify(samples=mate_samples,filter_sites=False)
	# Write sample map of ancestry for samples
	mate_sample_map = _write_sample_map(tree_mate,N_MATE,N_MATE)
	# Output founder tree and sample map to file
	tree_mate.dump(prefix+"trees/tree_mate.hdf")
	mate_sample_map.to_csv(prefix+"trees/sample_map_mate.txt",header=False,sep="\t",index=False)
	# Write input files needed for admixture simulation
	with gzip.open(prefix+"admixed_data/input/ceu_yri_genos.vcf.gz", "wt") as f:
		tree_mate.write_vcf(f,ploidy=2,contig_id=chrom)
		mate_sample_map.iloc[:,:2].to_csv(prefix+"admixed_data/input/ceu_yri_map.txt",sep="\t",header=False,index=False)
	return mate_samples

def _write_sample_map(tree, N_CEU, N_YRI):
    """
    Write ancestry of samples

    Parameters
    ----------
    tree : msprime.TreeSequence
	simulated population genotypes and samples from msprime
    N_CEU : int
	Number of samples of European ancestry
    N_YRI : int
	Number of samples of African ancestry

    Returns
    -------
    pd.DataFrame
	Table with sample IDs, haplotype positions, and ancestry 
    """

    # Codes that relate to each ancestry group
    pop_dict = {0:"YRI",1:"CEU",2:"CHB"}
    pops,inds,hap1,hap2 = [],[],[],[]
    count=0 # sample id tracker
    # For each sample get the ancestry
    for i in range(0,2*(N_CEU+N_YRI),2):
            pops.append(pop_dict.get(tree.get_population(i))) # ancestry of individual
            inds.append("msp_"+str(count))	# id of individual
            hap1.append(i) # Position of haplotype 1
            hap2.append(i+1) # Position of haplotype 2
            count+=1 
    # Combine into dataframe
    df = pd.DataFrame(np.column_stack([pops,hap1,hap2]), index=inds)
    df = df.reset_index()
    df.columns = np.arange(0,4)
    return df

def _out_of_africa(N_CEU, N_YRI, N_CHB, rmap):
	"""
	This function is copied from the msprime documentation found here: 
	https://msprime.readthedocs.io/en/stable/tutorial.html#demography 
	

	It is used to simulate African, European, and Asian individuals based on the Out of 
	Africa model developed by Gutenkunst et al. from the HapMap data
	"""
	
    # First we set out the maximum likelihood values of the various parameters
    # given in Table 1.
	N_A = 7300
	N_B = 2100
	N_AF = 12300
	N_EU0 = 1000
	N_AS0 = 510
	# Times are provided in years, so we convert into generations.
	generation_time = 25
	T_AF = 220e3 / generation_time
	T_B = 140e3 / generation_time
	T_EU_AS = 21.2e3 / generation_time
	# We need to work out the starting (diploid) population sizes based on
	# the growth rates provided for these two populations
	r_EU = 0.004
	r_AS = 0.0055
	N_EU = N_EU0 / math.exp(-r_EU * T_EU_AS)
	N_AS = N_AS0 / math.exp(-r_AS * T_EU_AS)
	# Migration rates during the various epochs.
	m_AF_B = 25e-5
	m_AF_EU = 3e-5
	m_AF_AS = 1.9e-5
	m_EU_AS = 9.6e-5
	# Population IDs correspond to their indexes in the population
	# configuration array. Therefore, we have 0=YRI, 1=CEU and 2=CHB
	# initially.
	population_configurations = [
	    msprime.PopulationConfiguration(
	        sample_size=(2*N_YRI), initial_size=N_AF),
	    msprime.PopulationConfiguration(
	        sample_size=(2*N_CEU), initial_size=N_EU, growth_rate=r_EU),
	    msprime.PopulationConfiguration(
	        sample_size=(2*N_CHB), initial_size=N_AS, growth_rate=r_AS)
	]
	migration_matrix = [
	    [      0, m_AF_EU, m_AF_AS],
	    [m_AF_EU,       0, m_EU_AS],
	    [m_AF_AS, m_EU_AS,       0],
	]
	demographic_events = [
	    # CEU and CHB merge into B with rate changes at T_EU_AS
	    msprime.MassMigration(
	        time=T_EU_AS, source=2, destination=1, proportion=1.0),
	    msprime.MigrationRateChange(time=T_EU_AS, rate=0),
	    msprime.MigrationRateChange(
	        time=T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
	    msprime.MigrationRateChange(
	        time=T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
	    msprime.PopulationParametersChange(
	        time=T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1),
	    # Population B merges into YRI at T_B
	    msprime.MassMigration(
	        time=T_B, source=1, destination=0, proportion=1.0),
	    msprime.MigrationRateChange(time=T_B, rate=0), # added rate change initially missing
	    # Size changes to N_A at T_AF
	    msprime.PopulationParametersChange(
	        time=T_AF, initial_size=N_A, population_id=0)
	]

	tree = msprime.simulate(mutation_rate=2e-8,
	                        recombination_map=rmap,
	                        population_configurations=population_configurations,
	                        migration_matrix=migration_matrix,
	                        demographic_events=demographic_events)
	return tree
