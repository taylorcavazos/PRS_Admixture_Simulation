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
	default="genetic_map_GRCh37_chr22.txt")
parser.add_argument("--m",help="number of causal variants", type=int, default=1000)
parser.add_argument("--h2",help="heritability", type=float, default=0.67)
parser.add_argument("--iter", help="iteration number", type=int, default=1)

args = parser.parse_args()

N_CEU = 206000
N_YRI = 6000

print("Simulating main population for iteration = {}".format(args.iter))

if os.path.isfile("sim{}/trees/tree_all.hdf".format(args.iter)):
	main_tree = msprime.load("sim{}/trees/tree_all.hdf".format(args.iter))
	sample_map_all = pd.read_csv("sim{}/trees/sample_map_all.txt".format(args.iter),
	sep="\t",header=None)

else:
	main_tree, sample_map_all = simulate_out_of_afr(N_CEU, N_YRI, args.rmapFile)
	main_tree.dump("sim{}/trees/tree_all.hdf".format(args.iter))
	sample_map_all.to_csv("sim{}/trees/sample_map_all.txt".format(args.iter),
		header=False,sep="\t",index=False)


print("Number of sites = {}".format(main_tree.num_sites))

EUR_all = sample_map_all[sample_map_all.iloc[:,1]=="CEU"]
YRI_all = sample_map_all[sample_map_all.iloc[:,1]=="YRI"]

print("Selecting samples for mating")

EUR_mate = EUR_all.loc[np.random.choice(EUR_all.index,size=4000,replace=False)]
YRI_mate = YRI_all.loc[np.random.choice(YRI_all.index,size=4000,replace=False)]

ALL_mate = pd.concat([EUR_mate,YRI_mate])

mate_samples = []

for ind in ALL_mate.index:
    mate_samples.append(ALL_mate.loc[ind,2])
    mate_samples.append(ALL_mate.loc[ind,3])

print("Saving mating population tree")

tree_mate = main_tree.simplify(samples=mate_samples,filter_sites=False)
mate_sample_map = write_sample_map(tree_mate,4000,4000)

tree_mate.dump("sim{}/trees/tree_mate.hdf".format(args.iter))
mate_sample_map.to_csv("sim{}/trees/sample_map_MATE.txt".format(args.iter),header=False,sep="\t",index=False)

print("Saving non-mating population tree")

all_data = np.array(main_tree.samples()).astype(np.int32)
other_samps = np.delete(all_data,np.array(mate_samples).astype(np.int32))
tree_other = main_tree.simplify(samples = other_samps, filter_sites=False)

sample_map_other = write_sample_map(tree_other,202000,2000)


EUR_other = sample_map_other[sample_map_other.iloc[:,1]=="CEU"]
YRI_other = sample_map_other[sample_map_other.iloc[:,1]=="YRI"]

print("Saving YRI LD population tree")

YRI_other_LD = []

for ind in YRI_other.index:
	YRI_other_LD.append(YRI_other.loc[ind,2])
	YRI_other_LD.append(YRI_other.loc[ind,3])

tree_YRI_LD = tree_other.simplify(samples = YRI_other_LD, filter_sites=False)
tree_YRI_LD.dump("sim{}/trees/tree_YRI_LD_nofilt.hdf".format(args.iter))


EUR_other_LD = []

for ind in np.random.choice(EUR_other.index, size=2000, replace=False):
	EUR_other_LD.append(EUR_other.loc[ind,2])
	EUR_other_LD.append(EUR_other.loc[ind,3])

samps_b4_gwas = np.array(tree_other.samples()).astype(np.int32)

EUR_other_GWAS = np.delete(samps_b4_gwas,np.array(EUR_other_LD+YRI_other_LD).astype(np.int32))

tree_EUR_LD = tree_other.simplify(samples=EUR_other_LD, filter_sites = False)
tree_EUR_GWAS = tree_other.simplify(samples=EUR_other_GWAS, filter_sites=False)

print("Saving CEU LD and GWAS population trees")

tree_EUR_LD.dump("sim{}/trees/tree_CEU_LD_nofilt.hdf".format(args.iter))
tree_EUR_GWAS.dump("sim{}/trees/tree_CEU_GWAS_nofilt.hdf".format(args.iter))

print("Writing inputs for admixture analysis")
with gzip.open("sim{}/admixed_data/input/ceu_yri_genos.vcf.gz".format(args.iter), "wt") as f:
    tree_mate.write_vcf(f,ploidy=2,contig_id="22")
mate_sample_map.iloc[:,:2].to_csv("sim{}/admixed_data/input/ceu_yri_map.txt",sep="\t",header=False,index=False)
