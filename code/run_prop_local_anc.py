import matplotlib.pyplot as plt
import seaborn as sns
import sys
import gzip

sys.path.insert(0,"/Users/taylorcavazos/repos/Local_Ancestry_PRS/code/")
sys.path.insert(0,"/Users/taylorcavazos/Documents/Prelim_Quals/Aim1")

from sim_out_of_africa import *
from output_true_prs import *

path_tree = "/Users/taylorcavazos/repos/Local_Ancestry_PRS/data/trees/tree_sub_CEU_1.95e5.hdf"
n_sites = msprime.load(path_tree).num_sites

with open("/Users/taylorcavazos/repos/other_tools/admixture-data/output/admix_afr_amer.result") as f:
    pbar = tqdm.tqdm(total=n_sites+1)
    ind = 0
    for line in f:
        if ind == 0:
            sample_haps = line.split("\t")[2:]
            samples = [sample_haps[i].split(".")[0] for i in range(0,len(sample_haps),2)]
            anc_df = pd.DataFrame(index=samples,columns=["Prop_CEU","Prop_YRI"])
            counts_CEU = np.zeros(len(samples))
            counts_YRI = np.zeros(len(samples))
        else:
            haplo_anc = np.array(line.split("\t")[2:]).astype(int)
            YRI_arr = haplo_anc-1
            line_counts_YRI = np.add.reduceat(YRI_arr, np.arange(0, len(YRI_arr), 2))
            
            CEU_arr = np.absolute(1-YRI_arr)
            line_counts_CEU = np.add.reduceat(CEU_arr, np.arange(0, len(CEU_arr), 2))
            
            counts_CEU = counts_CEU+line_counts_CEU
            counts_YRI = counts_YRI+line_counts_YRI
        ind+=1
        pbar.update(1)
    anc_df["Prop_CEU"] = counts_CEU/(2*(ind-1))
    anc_df["Prop_YRI"] = counts_YRI/(2*(ind-1))
    anc_df.to_csv("/Users/taylorcavazos/repos/other_tools/admixture-data/output/admix_afr_amer.prop.anc",sep="\t")
