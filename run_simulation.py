import simulation as SIM
from config import *
import argparse
import os

def main(sim):
	# Create directories needed for simulation
	os.system("mkdir -p output/sim{}".format(sim)+"/{trees,admixed_data/{input,output},true_prs,emp_prs,plots}")
	
	# Part 1: Simulate populations
	if os.path.isfile("output/sim{}/trees/tree_all.hdf".format(sim)):
		print("Population for iteration={} exists".format(sim))
		print("If you would like to overwrite, remove output/sim{}/trees/tree_all.hdf".format(sim))
	else:
		print("Simulating populations for iteration={}".format(sim))
		# simulate_populations(N_CEU, N_YRI, N_MATE, N_ADMIX, rmap_file, prefix="../output/sim{}/".format(sim))
		SIM.simulate_populations(110, 110, 10, 50, rmap_file, prefix="output/sim{}/".format(sim))

	# Part 2: Construct true polygenic risk scores and split into case/control
	## optional parameters which can be modified: m, h2

	if os.path.isfile("output/sim{}/true_prs/prs_m_{}_h2_{}.hdf5)".format(sim,M,H2)):
		print("True PRS for iteration={} exists".format(sim))
		print("If you would like to overwrite, remove output/sim{}/true_prs/prs_m_{}_h2_{}.hdf".format(sim,M,H2))
	else:
		print("Simulating true PRS for iteration={}".format(sim))
	# simulate_true_prs(M, H2, N_ADMIX, prefix="../output/sim{}/".format(sim))
		SIM.simulate_true_prs(M, H2, 50, prefix="output/sim{}/".format(sim))

	# Part 3: Compute empirical polygenic risk scores
	## optional parameters which can be modified: p-value, r2, weighting, snp-selection (future)

parser = argparse.ArgumentParser(description="Simulation of population trees")
parser.add_argument("--sim",help="population identifier", type=str, default="1")

args = parser.parse_args()
main(args.sim)