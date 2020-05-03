# PRS Admixture Simulation 

A repository for exploring PRS weighting and snp selection strategies for generalizing polygenic risk scores in admixed individuals.


These functions simulate European, African, and Admixed individuals using [msprime](https://github.com/tskit-dev/msprime) and rfmix's [simulate](https://github.com/tskit-dev/msprime) function. Then a true polygenic risk score is constructed as described by [Martin et. al. 2017, AJHG](https://www.ncbi.nlm.nih.gov/pubmed/28366442). GWAS estimated polygenic risk scores are constructed by P+T with SNPs selected based on summary statistics from a European or African population of a Fixed-Effects meta of both. SNPs are weighted using the same stragegies or by using weights specific to the local ancestry at a given PRS locus.

## Getting Started 
To run the functions described above follow the below instructions:
```
# Main package installation
git clone https://github.com/taylorcavazos/Local_Ancestry_PRS.git
cd Local_Ancestry_PRS
pip install requirements.txt

# Rfmix simulate installation and set-up
git clone https://github.com/slowkoni/rfmix.git
cd rfmix
autoreconf --force --install
./configure
make
cd ..
mv rfmix/simulate simulation/simulate-admixed
rm -rf rfmix
```

## Simulation parameters  
All parameters from the simulation can be obtained by `python run_simulation.py --help`. Output provided below:
```
Functions for simulating European, African, and Admixed populations and
testing PRS building strategies for better generalization to diverse
populations. Additional parameters can be adjusted in the config.py file

optional arguments:
  -h, --help            show this help message and exit
  --sim SIM             Population identifier. Must give unique value if you
                        want to run the simulation multiple times and retain
                        outputs.
  --m M                 # of causal variants to assume
  --h2 H2               heritability to assume
  --snp_weighting {ceu,yri,la,meta}
                        Weighting strategy for PRS building. Can use weights
                        from European (ceu) or African (yri) GWAS as well as
                        weights from a fixed-effects meta analysis of both
                        GWAS (meta) or local-ancestry specific weights (la).
  --snp_selection {ceu,yri,meta}
                        SNP selection strategy for PRS building. Choice
                        between using significant SNPs from a European (ceu)
                        or African (yri) GWAS.
  --pvalue PVALUE       pvalue cutoff to be used for LD clumping
  --ld_r2 LD_R2         r2 cutoff to be used for LD clumping
  --decrease_samples_yri DECREASE_SAMPLES_YRI
                        # of cases used in YRI analysis to represent the lack
                        of non-European data
  --output_dir OUTPUT_DIR
                        location for output data to be written
```

## Example run  
A possible simulation run is shown below:
```
python run_simulation.py --sim 1 --snp_selection ceu --snp_weighting meta
```



