import pandas as pd
import numpy as np
import gzip
import multiprocessing as mp
import argparse
import h5py

def main(f, m, h2):
    G = f['G'][()]

    e = np.random.normal(loc=0, scale=(1-h2), size=int(G.shape[0]))
    Ze = (e - np.mean(e))/np.std(e)
    E = np.sqrt(1-h2)*Ze

    labels = f['labels'][()].astype(str)
    print("Sample Breakdown")
    print("----------------")
    print("Number of individuals = {}".format(len(labels)))
    
    ceu_inds, admix_inds = [],[]
    for i in range(len(labels)):
        if "msp" in labels[i]: ceu_inds.append(i)
        else: admix_inds.append(i)
    ceu_inds = np.array(ceu_inds)
    admix_inds = np.array(admix_inds)
    ceu_G_plus_E = (G+E)[ceu_inds]

    print("Number of CEU individuals = {}".format(len(ceu_inds)))
    print("Number of ADMIX individuals = {}".format(len(admix_inds)))

    num_case = int(len(ceu_inds)*0.05)
    print("Number of expected cases/controls = {}".format(num_case))

    sorted_risk = np.argsort(ceu_G_plus_E)[::-1]
    labels_case = ceu_inds[sorted_risk[:num_case]]

    print("Number of cases found = {}".format(len(labels_case)))
    labels_control = ceu_inds[sorted_risk[num_case:]]

    train_controls = np.random.choice(labels_control, size=num_case, replace=False)
    print("Number of controls found = {}".format(len(train_controls)))
    
    remainder = [ind for ind in labels_control if ind not in train_controls]
    testing = np.random.choice(remainder, size=int(num_case/2), replace=False)
    testing_w_admix = np.append(testing, admix_inds)

    print("Number of testing samples = {}".format(len(testing_w_admix)))
    print("----------------")
    with h5py.File('true_prs/train_test_m_{}_h2_{}.hdf5'.format(m,h2), 'w') as f:
        f.create_dataset("train_cases",labels_case.shape,dtype=int,data=labels_case)
        f.create_dataset("train_controls",train_controls.shape,dtype=int,data=train_controls)
        f.create_dataset("test_data",testing_w_admix.shape,dtype=int,data=testing_w_admix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculates empirical prs")
    parser.add_argument("--m",help="number of causal variants", type=int, default=1000)
    parser.add_argument("--h2",help="heritability", type=float, default=0.67)

    args = parser.parse_args()
    true_prs_f = h5py.File("true_prs/prs_m_{}_h2_{}.hdf5".format(args.m,args.h2))

    main(true_prs_f, args.m, args.h2)