import simulation as SIM
from config import *
import argparse
import os

def main(sim):
	# PART 0: CREATE DIRECTORIES NEEDED FOR SIMULATION
	os.system("mkdir -p output/sim{}".format(sim)+"/{trees,admixed_data/{input,output},true_prs,emp_prs,plots,summary}")
	
	# PART 1: SIMULATE POPULATIONS

	if os.path.isfile("output/sim{}/trees/tree_all.hdf".format(sim)):
		print("\nPopulation for iteration={} exists".format(sim))
		print("If you would like to overwrite, remove output/sim{}/trees/tree_all.hdf".format(sim))
	else:
		print("\nSimulating populations for iteration={}".format(sim))
		# SIM.simulate_populations(N_CEU, N_YRI, N_MATE, N_ADMIX, rmap_file, prefix="../output/sim{}/".format(sim))
		SIM.simulate_populations(110, 110, 10, 50, rmap_file, prefix="output/sim{}/".format(sim))

	# PART 2: CONSTRUCT TRUE POLYGENIC RISK SCORES AND SPLIT INTO CASE/CONTROL

	if os.path.isfile("output/sim{}/true_prs/prs_m_{}_h2_{}.hdf5".format(sim,M,H2)):
		print("\nTrue PRS for iteration={} exists".format(sim))
		print("If you would like to overwrite, remove output/sim{}/true_prs/prs_m_{}_h2_{}.hdf5".format(sim,M,H2))
	else:
		print("\nSimulating true PRS for iteration={}".format(sim))
		# SIM.simulate_true_prs(M, H2, N_ADMIX, prefix="../output/sim{}/".format(sim))
		SIM.simulate_true_prs(M, H2, 50, prefix="output/sim{}/".format(sim))

	# PART 3: COMPUTE EMPIRICAL POLYGENIC RISK SCORES
	## optional parameters which can be modified: p-value, r2, weighting, snp-selection (future)

parser = argparse.ArgumentParser(description="Simulation of population trees")
parser.add_argument("--sim",help="population identifier", type=str, default="1")

args = parser.parse_args()
main(args.sim)