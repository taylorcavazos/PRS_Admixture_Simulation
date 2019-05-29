import matplotlib.pyplot as plt
import seaborn as sns
import sys
import gzip

sys.path.insert(0,"/Users/taylorcavazos/repos/Local_Ancestry_PRS/code/")
sys.path.insert(0,"/Users/taylorcavazos/Documents/Prelim_Quals/Aim1")

from sim_out_of_africa import *
from output_true_prs import *

import output_true_prs

m = 1000
h2 = 0.67
path_tree = "/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/trees/tree_sub_CEU_1.95e5.hdf"

output_true_prs.main(path_tree,m,h2,"/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/true_prs/prs_m_{}_h2_{}".format(m,h2),iters=100,
                     admix_vcf = "/Users/taylorcavazos/repos/other_tools/admixture-data/output/admix_afr_amer.query.vcf")
