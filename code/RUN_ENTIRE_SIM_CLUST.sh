#!/bin/bash
#
#PBS -N Run_Local_Ancestry_Sim
#PBS -S /bin/bash
#PBS -o /wittelab/data2/tcavazos/
#PBS -e /wittelab/data2/tcavazos/
#PBS -l nodes=1:ppn=8
#PBS -l vmem=50gb,mem=50gb
#PBS -t 1-25
#PBS -m abe
#PBS -M taycav2@gmail.com

uname -a
date
echo "PBS Job ID: $PBS_JOBID"
echo "PBS Task ID: $PBS_ARRAYID"

cd /Users/taylorcavazos/repos/Local_Ancestry_PRS/data

mkdir -p sim$PBS_ARRAYID/{trees,admixed_data/{input,output},true_prs,emp_prs}
cd sim$PBS_ARRAYID


################################# DATA SIMULATION AND PREP #################################
python ../../code/construct_sim_trees.py --iter ${PBS_ARRAYID} \
		--numCEU 2200 \
		--numYRI 200 \
		--numMATE 100 \
		--numLD 100

echo "Simulating admixed individuals"

python2.7 ../../code/admixture-simulation/do-admixture-simulation.py \
	--input-vcf admixed_data/input/ceu_yri_genos.vcf.gz \
	--sample-map admixed_data/input/ceu_yri_map.txt \
	--n-output 50 \
	--n-generations 8 \
	--chromosome 22 \
	--genetic-map ../genetic_map_GRCh37_chr22_fix.txt \
	--output-basename admixed_data/output/admix_afr_amer

echo "Calculating proportion of local ancestry"

python ../../code/run_prop_local_anc.py

################################# CREATE TRUE PRS #################################

python ../../code/output_true_prs.py --m 1000 --h2 0.67 --numADMIX 50 --iters 2

################################# CHOOSE CASES/CONTROLS #################################

echo "Splitting cases and controls"
python ../../code/split_case_control.py --m 1000 --h2 0.67

################################# CALCULATE SUMMARY STATISTICS #################################

echo "Computing summary statistics"
python ../../code/run_compute_sum_stats.py --m 1000 --h2 0.67

################################# CREATE EMPIRICAL PRS #################################

echo "Running empricial prs"
python ../../code/output_emp_prs.py --m 1000 --h2 0.67
	

# EXTEND TO LOOP THROUGH H2 AND M
	# for m in {200, 500, 1000}; do
	# 	for h2 in {0.33, 0.50, 0.67}; do

	# 	done

	# done