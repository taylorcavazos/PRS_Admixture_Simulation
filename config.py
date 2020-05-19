"""
Global simulation parameters
"""

# Population sizes
N_CEU = 201000
N_YRI = 201000
N_MATE = 1000
N_ADMIX = 5000

# Chromosome recombination file
rmap_file = "required_data/genetic_map_GRCh37_chr20.txt"

# True polygenic risk score
M=1000 # causal variants
H2=0.5 # heritability

# Empirical polygenic risk score
P=0.01 # p-value cutoff
R2=0.2 # LD threshold
