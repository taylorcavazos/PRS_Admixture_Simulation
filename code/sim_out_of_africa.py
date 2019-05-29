# These functions are used to simulate homeogeneous European and African
# samples based on Gutenkunst et als. out of africa model using msprime

import msprime
import math
import pandas as pd
import numpy as np

def out_of_africa(N_CEU, N_YRI, rmap, N_CHB=0, 
	prefix="/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/trees/", 
	chrom=22):
	"""
	This function is copied from the msprime documentation. It is used to
	simulate African, European, and Asian individuals based on the Out of 
	Africa model developed by Gutenkunst et al. from the HapMap data

	Parameters
	----------
	N_CEU : int
		Number of samples of European ancestry
	N_YRI : int
		Number of samples of African ancestry
	rmap : msprime.simulations.RecombinationMap
		Recombination map for a reference chromosome
	N_CHB : int, optional
		Number of samples of Asian ancestry
	prefix : str, optional
		directory for saving hdf of tree to
	chrom : str, optional
		Chromosome number of rmap file

	Returns
	-------
	tskit.trees.TreeSequence
		Tree object from which sample haplotypes can be
		accessed
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
	        sample_size=N_CHB, initial_size=N_AS, growth_rate=r_AS)
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
	    # Size changes to N_A at T_AF
	    msprime.PopulationParametersChange(
	        time=T_AF, initial_size=N_A, population_id=0)
	]

	tree = msprime.simulate(mutation_rate=2e-8,
	                        recombination_map=rmap,
	                        population_configurations=population_configurations,
	                        migration_matrix=migration_matrix,
	                        demographic_events=demographic_events)
	tree.dump(prefix+"tree_YRI_5e4_CEU_2e6_chr{}.hdf5".format(str(chrom)))
	return tree

def write_sample_map(tree,outdir, N_CEU, N_YRI):
	"""
	Write file with ids and population for each sample

	Parameters
	----------
	tree : msprime.simulations.RecombinationMap
		Tree sequence that is output of msprime.simulate(...)
	outfile : str
		Output file for writing the sample map
	"""
	pop_dict = {0:"YRI",1:"CEU",2:"CHB"}
	pops,inds,hap1,hap2 = [],[],[],[]
	count=0
	for i in range(0,2*(N_CEU+N_YRI),2):
		pops.append(pop_dict.get(tree.get_population(i)))
		inds.append("msp_"+str(count))
		#hap1.append(i)
		#hap2.append(i+1)
		count+=1
	pd.DataFrame(pops, index=inds).to_csv(outdir+"CEU_YRI_sample_map.txt",header=False,sep="\t")
	return

def simulate_out_of_afr(N_CEU, N_YRI, rmap, outdir, N_CHB=0, chrom=22):
	"""
	Function to simulate European and African homogeneous populations based on
	msprime's out_of_africa model. Outputs vcf of sample genotypes and sample map
	for input into admixture simulations.

	Parameters
	----------
	N_CEU : int
		Number of samples of European ancestry
	N_YRI : int
		Number of samples of African ancestry
	rmap : msprime.simulations.RecombinationMap
		Recombination map for a reference chromosome
	outdir : str
		directory to write data outputs to
	N_CHB : int, optional
		Number of samples of Asian ancestry
	chrom : str, optional
		Chromosome number of rmap file

	"""
	print("Simulating populations with msprime")
	#tree = out_of_africa(N_CEU, N_YRI, rmap)
	tree = msprime.load("/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/trees/tree_YRI_5e4_CEU_2e6_chr22.hdf5")
	#print("Writing genotypes to vcf")
	#tree.write_vcf(open(outdir+"YRI_CEU_chr{}_1e4.vcf".format(chrom),"w"), ploidy=2, contig_id=str(chrom))
	print("Writing sample map")
	write_sample_map(tree, outdir, N_CEU, N_YRI)

