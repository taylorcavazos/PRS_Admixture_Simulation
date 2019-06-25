"""
 This will be the first step in the simulation pipeline. Here we want to simulate our trees 
(1 with admixed individuals, 1 with all others), choose YRI individuals for LD panel, choose CEU
individuals for LD panel, 
"""

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import gzip
import multiprocessing as mp
import argparse
import os.path

from sim_out_of_africa import *

parser = argparse.ArgumentParser(description="Simulation of population trees")
parser.add_argument("--rmapFile",help="number of causal variants", type=str, 
	default="../genetic_map_GRCh37_chr22.txt")
parser.add_argument("--m",help="number of causal variants", type=int, default=1000)
parser.add_argument("--h2",help="heritability", type=float, default=0.67)
parser.add_argument("--iter", help="iteration number", type=int, default=1)

parser.add_argument("--numCEU",help="number of Europeans", type=int, default=203000)
parser.add_argument("--numYRI",help="number of Africans", type=int, default=3000)
parser.add_argument("--numMATE", help="number to use for mating", type=int, default=1000)
parser.add_argument("--numLD", help="number to use for LD reference", type=int, default=2000)

args = parser.parse_args()

N_CEU = args.numCEU
N_YRI = args.numYRI
N_MATE = args.numMATE
N_LD = args.numLD

print("Simulating main population for iteration = {}".format(args.iter))
print("Population Breakdown")
print("--------------------")
print("Number CEU: {}".format(N_CEU))
print("Number YRI: {}".format(N_YRI))
print("Number for mating: {}".format(N_MATE))
print("Number for LD reference: {}".format(N_LD))
print("--------------------")

if os.path.isfile("trees/tree_all.hdf"):
	main_tree = msprime.load("trees/tree_all.hdf")
	sample_map_all = pd.read_csv("trees/sample_map_all.txt",
	sep="\t",header=None)

else:
	main_tree, sample_map_all = simulate_out_of_afr(N_CEU, N_YRI, args.rmapFile)
	main_tree.dump("trees/tree_all.hdf")
	sample_map_all.to_csv("trees/sample_map_all.txt",
		header=False,sep="\t",index=False)

print("Number of samples = {}".format(main_tree.num_samples/2))
print("Number of sites = {}".format(main_tree.num_sites))

EUR_all = sample_map_all[sample_map_all.iloc[:,1]=="CEU"]
YRI_all = sample_map_all[sample_map_all.iloc[:,1]=="YRI"]


print("Selecting samples for mating")

EUR_mate = EUR_all.loc[np.random.choice(EUR_all.index,size=N_MATE,replace=False)]
YRI_mate = YRI_all.loc[np.random.choice(YRI_all.index,size=N_MATE,replace=False)]

ALL_mate = pd.concat([EUR_mate,YRI_mate])

mate_samples = []

for ind in ALL_mate.index:
    mate_samples.append(ALL_mate.loc[ind,2])
    mate_samples.append(ALL_mate.loc[ind,3])

mate_samples = np.array(mate_samples).astype(np.int32)

print("Saving mating population tree")

tree_mate = main_tree.simplify(samples=mate_samples,filter_sites=False)
mate_sample_map = write_sample_map(tree_mate,N_MATE,N_MATE)

tree_mate.dump("trees/tree_mate.hdf")
mate_sample_map.to_csv("trees/sample_map_MATE.txt",header=False,sep="\t",index=False)

print("Saving non-mating population tree")

all_data = np.array(main_tree.samples()).astype(np.int32)
other_samps = [ind for ind in all_data if ind not in mate_samples]
tree_other = main_tree.simplify(samples = other_samps, filter_sites=False)
sample_map_other = write_sample_map(tree_other,N_CEU-N_MATE,N_YRI-N_MATE)


EUR_other = sample_map_other[sample_map_other.iloc[:,1]=="CEU"]
YRI_other = sample_map_other[sample_map_other.iloc[:,1]=="YRI"]

print("Saving YRI LD population tree")

YRI_other_LD = []

for ind in YRI_other.index:
	YRI_other_LD.append(YRI_other.loc[ind,2])
	YRI_other_LD.append(YRI_other.loc[ind,3])

tree_YRI_LD = tree_other.simplify(samples = YRI_other_LD, filter_sites=False)
tree_YRI_LD.dump("trees/tree_YRI_LD_nofilt.hdf")


EUR_other_LD = []

inds = np.random.choice(EUR_other.index, size=N_LD, replace=False)
for ind in inds:
	EUR_other_LD.append(EUR_other.loc[ind,2])
	EUR_other_LD.append(EUR_other.loc[ind,3])

samps_b4_gwas = tree_other.samples()

all_LD = np.array(EUR_other_LD+YRI_other_LD).astype(int)


EUR_other_GWAS = np.array([samp for samp in samps_b4_gwas if samp not in all_LD])
tree_EUR_LD = tree_other.simplify(samples=EUR_other_LD, filter_sites = False)
tree_EUR_GWAS = tree_other.simplify(samples=EUR_other_GWAS, filter_sites=False)

print("Saving CEU LD and GWAS population trees")

tree_EUR_LD.dump("trees/tree_CEU_LD_nofilt.hdf")
tree_EUR_GWAS.dump("trees/tree_CEU_GWAS_nofilt.hdf")

print("Writing inputs for admixture analysis")
with gzip.open("admixed_data/input/ceu_yri_genos.vcf.gz", "wt") as f:
    tree_mate.write_vcf(f,ploidy=2,contig_id="22")
mate_sample_map.iloc[:,:2].to_csv("admixed_data/input/ceu_yri_map.txt",sep="\t",header=False,index=False)
