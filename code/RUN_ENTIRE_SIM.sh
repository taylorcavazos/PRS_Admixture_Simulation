cd /Users/taylorcavazos/repos/Local_Ancestry_PRS/data

for it in {1..1}; do
	mkdir sim$it
	mkdir sim$it/trees
	mkdir sim$it/admixed_data
	mkdir sim$it/admixed_data/input
	mkdir sim$it/admixed_data/output
	mkdir sim$it/true_prs
	mkdir sim$it/emp_prs

	cd sim$it


################################# DATA SIMULATION AND PREP #################################
	python /Users/taylorcavazos/repos/Local_Ancestry_PRS/code/construct_sim_trees.py --iter $it

	echo "Simulating admixed individuals"

	python2.7 /Users/taylorcavazos/repos/other_tools/admixture-simulation/do-admixture-simulation.py \
		--input-vcf admixed_data/input/ceu_yri_genos.vcf.gz \
		--sample-map admixed_data/input/ceu_yri_map.txt \
		--n-output 4000 \
		--parent-percent 1 \
		--n-generations 8 \
		--chromosome 22 \
		--genetic-map genetic_map_GRCh37_chr22.txt \
		--output-basename admixed_data/output/admix_afr_amer

# To do: Calculate ancestry proportions


################################# CREATE TRUE PRS #################################


################################# CREATE EMPIRICAL PRS #################################

done