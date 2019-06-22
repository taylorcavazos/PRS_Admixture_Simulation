cd /Users/taylorcavazos/repos/Local_Ancestry_PRS/data

for it in {1..1}; do
	mkdir -p sim$it/{trees,admixed_data/{input,output},true_prs,emp_prs}
	cd sim$it


################################# DATA SIMULATION AND PREP #################################
	# python ../../code/construct_sim_trees.py --iter $it \
	# 		--numCEU 2000 \
	# 		--numYRI 200 \
	# 		--numMATE 100 \
	# 		--numLD 100

	# echo "Simulating admixed individuals"

	# python2.7 ../../code/admixture-simulation/do-admixture-simulation.py \
	# 	--input-vcf admixed_data/input/ceu_yri_genos.vcf.gz \
	# 	--sample-map admixed_data/input/ceu_yri_map.txt \
	# 	--n-output 50 \
	# 	--n-generations 8 \
	# 	--chromosome 22 \
	# 	--genetic-map ../genetic_map_GRCh37_chr22_fix.txt \
	# 	--output-basename admixed_data/output/admix_afr_amer

	# echo "Calculating proportion of local ancestry"

	# python ../../code/run_prop_local_anc.py

################################# CREATE TRUE PRS #################################

	python ../../code/output_true_prs.py --m 1000 --h2 0.67 --numADMIX 50 --iters 10

################################# CHOOSE CASES/CONTROLS #################################

################################# CREATE EMPIRICAL PRS #################################

done

# EXTEND TO LOOP THROUGH H2 AND M
	# for m in {200, 500, 1000}; do
	# 	for h2 in {0.33, 0.50, 0.67}; do

	# 	done

	# done