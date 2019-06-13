import matplotlib.pyplot as plt
import seaborn as sns
import sys
import gzip
import multiprocessing as mp
import argparse

sys.path.insert(0,"/Users/taylorcavazos/repos/Local_Ancestry_PRS/code/")

from sim_out_of_africa import *

parser = argparse.ArgumentParser(description="Simulation of population trees")
parser.add_argument("--m",help="number of causal variants", type=int, default=1000)
parser.add_argument("--h2",help="heritability", type=float, default=0.67)
parser.add_argument("--iter", help="iteration number", type=int, default=1)

args = parser.parse_args()

N_CEU = 208000
N_YRI = 8000

