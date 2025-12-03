#!/usr/bin/env python 

import numpy as np
import os
import pickle
import warnings
import pandas as pd
import argparse

# suppress warning messages to not clog stdout
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Generate a Metadata Pickle of a \
                                     Chromatography Setup (RP or HILIC).")
    parser.add_argument('--data_directory', help="Specify the processed data\
                        directory.", required=True)
    parser.add_argument('--tanaka', help="Specify the tanaka file",
                        default="/projects/bgmp/shared/groups/2025/deepmetab/shared/tanaka_database.tsv")
    parser.add_argument('--hsmb', help= "Specify the hsmb file",
                        default="/projects/bgmp/shared/groups/2025/deepmetab/shared/hsmb_database.tsv")
    parser.add_argument('-o', '--output', help= "Specify an output directory",required=True)
    parser.add_argument('-of', '--output_file', help="Specify Output File",
                        default="dom_the_goat.pickle")
    
    return parser.parse_args()


args = get_args()
# created a new directory to store processed data
#path = r'/projects/bgmp/shared/groups/2025/deepmetab/dnhem/deep_metab/dom_processed_data'#'/home/cmkstien/Desktop/rt_backup/RepoRT/processed_data'

# Store path to the processed data
path = args.data_directory
output = args.output
output_file = args.output_file

def get_inflections(pB, times):
    max_ind = np.argmax(pB) ## this is used to stop before the max value (basically ignore points after the gradient resets)
    slopes = []
    inflections = []
    for i in range(1, len(pB)):
        slope = (pB[i]-pB[i-1])/(times[i]-times[i-1])
        slopes.append(slope)
        if pB[i] != pB[i-1]:
            if i <= max_ind:
                inflections.append((times[i-1], pB[i-1]))
                inflections.append((times[i], pB[i]))
                inflections = list(set(inflections))
        
    return inflections

# Filtered List? Not present in current directory
# consider removing this part
# include a try statement here
try:
    with open(r'/home/cmkstien/Desktop/RT_data/filtered_not_ret_Jun24/filtered_list_june24.csv', 'r') as file:
        filter_list = file.readlines()
        filter_list = [x.strip() for x in filter_list]
        # print(filter_list)
except:
    print("Filtered List Not Present, initiliazing an empty one")
    filter_list = [] # initiliaze as empty
t_rt = 0

data = []

params_header =['company_name', 'usp_code', 'col_length', 'col_innerdiam', 'col_part_size', 'temp', 'col_fl', 'col_dead', 'HPLC_type','A_solv', 'B_solv', 'time1', 'grad1', 'time2', 'grad2', 'time3', 'grad3', 'time4', 'grad4',  \
            'A_pH', 'B_pH', 'A_start', 'A_end', 'B_start', 'B_end', \
            'eluent.A.formic', 'eluent.A.formic.unit', 'eluent.A.acetic', 'eluent.A.acetic.unit','eluent.A.trifluoroacetic', 'eluent.A.trifluoroacetic.unit','eluent.A.phosphor', 'eluent.A.phosphor.unit','eluent.A.nh4ac','eluent.A.nh4ac.unit', \
            'eluent.A.nh4form','eluent.A.nh4form.unit','eluent.A.nh4carb', \
            'eluent.A.nh4carb.unit','eluent.A.nh4bicarb','eluent.A.nh4bicarb.unit', \
            'eluent.A.nh4f','eluent.A.nh4f.unit','eluent.A.nh4oh', \
            'eluent.A.nh4oh.unit','eluent.A.trieth','eluent.A.trieth.unit', \
            'eluent.A.triprop','eluent.A.triprop.unit','eluent.A.tribut', \
            'eluent.A.tribut.unit','eluent.A.nndimethylhex', \
            'eluent.A.nndimethylhex.unit','eluent.A.medronic', \
            'eluent.A.medronic.unit',\
            
            'eluent.B.formic', 'eluent.B.formic.unit', 'eluent.B.acetic', 'eluent.B.acetic.unit','eluent.B.trifluoroacetic', 'eluent.B.trifluoroacetic.unit','eluent.B.phosphor', 'eluent.B.phosphor.unit','eluent.B.nh4ac','eluent.B.nh4ac.unit', \
            'eluent.B.nh4form','eluent.B.nh4form.unit','eluent.B.nh4carb', \
            'eluent.B.nh4carb.unit','eluent.B.nh4bicarb','eluent.B.nh4bicarb.unit', \
            'eluent.B.nh4f','eluent.B.nh4f.unit','eluent.B.nh4oh', \
            'eluent.B.nh4oh.unit','eluent.B.trieth','eluent.B.trieth.unit', \
            'eluent.B.triprop','eluent.B.triprop.unit','eluent.B.tribut', \
            'eluent.B.tribut.unit','eluent.B.nndimethylhex', \
            'eluent.B.nndimethylhex.unit','eluent.B.medronic', \
            'eluent.B.medronic.unit', \
            'kPB', 'αCH2', 'αT/O', 'αC/P', 'αB/P', 'αB/P.1', 'particle size', 'pore size', \
            'H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)', 'EB retention factor']
        
header = ['dir','name', 'smiles', 'rt']
col_dict = {}
col_dict['1'] = params_header
t= 0

# modify tanaka path with our directory
tanaka_path = args.tanaka#r'/projects/bgmp/shared/groups/2025/deepmetab/shared/tanaka_database.tsv'#r'/home/cmkstien/Desktop/rt_backup/RepoRT/resources/tanaka_database/tanaka_database.tsv'
tanaka_data = np.loadtxt(tanaka_path, delimiter='\t', dtype='str')

# modify hsmb path with our directory
hsmb_path = args.hsmb#r'/projects/bgmp/shared/groups/2025/deepmetab/shared/hsmb_database.tsv'#r'/home/cmkstien/Desktop/rt_backup/RepoRT/resources/hsm_database/hsm_database.tsv'
hsmb_data = np.loadtxt(hsmb_path, delimiter='\t', dtype='str')

hsmb_dict = {}
# print(hsmb_data[0][:])
for i in hsmb_data[1:]:
    hsmb_dict[i[0]] = i[5:-2]

tanaka_dict = {}
for i in tanaka_data[1:]:
    tanaka_dict[i[0]] = i[3:-1]
n_rm = 0
removedRT = []

# Make an output directory, wont raise error if it already exists
os.makedirs(output, exist_ok=True)

# walk through the processed data set
# expects gradient.tsv, metadata.tsv and info.tsv [future to do: try and except if files missing]
for root, dirs, files in os.walk(path):
    for dir in dirs:
        if dir in filter_list:
            continue
        grad_n = dir + "_gradient.tsv"
        grad_path = os.path.join(path, dir, grad_n)

        meta_n = dir + "_metadata.tsv"        
        meta_path = os.path.join(path, dir, meta_n)

        info_n = dir + "_info.tsv"
        info_path = os.path.join(path, dir, info_n)

        try:
            meta_data = np.loadtxt(meta_path, delimiter='\t', dtype='str')
        except:
            with open(meta_path, 'r') as file:
                lines = file.readlines()
            # print(meta_path)
            with open(meta_path, 'w') as file:
                for line in lines:
                    modified_line = line.replace('/%', 'percent')
                    file.write(modified_line)

        meta_data = np.loadtxt(meta_path, delimiter='\t', dtype='str')
        column_name = meta_data[1, 1]
       # print(column_name)
        
        col_string = column_name.split(' ')

        company_name = col_string[0]
        col_type = col_string[-2:]

        usp_code = meta_data[1, 2]

        col_length = meta_data[1, 3]
        if col_length == '':
            col_length = 0
        
        col_innerdiam = meta_data[1, 4] ## this is valuable

        col_part_size = meta_data[1, 5]
        if col_part_size == '':
            col_part_size = 0


        temp = meta_data[1, 6]
        col_fl = meta_data[1, 7]
        col_dead = meta_data[1, 8]
        dead = [float(col_dead)]

        t_no = dead[0]

        if 'HILIC' in column_name or 'Amide' in column_name:
            print("HILIC BROKE MY SHIT")
            continue
        
        A_mobile = meta_data[:, 9:18]
        A_add = meta_data[:, 18:48]

        A_pH = meta_data[:, 48]
        A_start = meta_data[:, 169]
        A_end = meta_data[:, 173]

        B_mobile = meta_data[:, 49:58]
        B_add = meta_data[:, 58:88]

        B_pH = meta_data[:, 88]
        B_start = meta_data[:, 170]
        B_end = meta_data[:, 174]
        switched = False

        # Debug Print Statements
        #print(f'Bug 1: Numpy String Conversion Error to Float = {B_mobile[1:,:]}')
        B_mobile_df = pd.DataFrame(B_mobile[1:,:]).replace('', np.nan).astype(float)
        B_ind = np.nanargmax(B_mobile_df.values) # fixed bug here for B solv
        B_solv = B_mobile[0,B_ind]
        A_ind = np.argmax(np.asarray(A_mobile[1:,:20], dtype=float))
        A_solv = A_mobile[0,A_ind]

        info_data = np.loadtxt(info_path, delimiter='\t', dtype='str')
        HPLC_type = info_data[1,2]

        grad_data = np.loadtxt(grad_path, delimiter='\t', dtype='str')

        grad = grad_data[1:, :]
        grad = np.asarray(grad, dtype=np.float32)

        pA = np.asarray(grad[:, 1], dtype=float)
        pB = np.asarray(grad[:, 2], dtype=float)

        fl = grad[:,-1]
        times = grad[:,0]

        inflections = get_inflections(pB, times)

        t_crit = 0

        if not np.all(fl == fl[0]):

            prev = fl[0]

            c= 0
            rt_list = []
            for i in fl:
              
                if i != prev:
                    filter_list.append(dir)
                    t_crit = grad[c,0]
                    break

                c+=1


        if t_crit == 0: 
            t_crit = inflections[-1][0]

        grad_l = []
        grad_l.extend([str(times[0]), str(pB[0])])
        c = 1
        count = 0
        t_prev = -1
        inflections.sort(key=lambda x: x[0])
        t_pB_max = inflections[-1][0]
        t_filter =  t_no + (0.01 * t_pB_max) ##  ## this is the trehold for compounds that are not retained on teh column

        while count < len(inflections):
            t = inflections[count][0]
            pB = inflections[count][1]

            if abs(t - t_prev) < 0.3: ## removing things that have very similar step sizes 
                removed = inflections.pop(count)

                continue
            t_prev = t
            count+=1


        for infl in inflections:
            grad_l.extend([str(infl[0]), str(infl[1])])
            c+=1

        pad = np.zeros((8-len(grad_l)))
        pad = np.asarray(pad, dtype=str)
        grad_l.extend(pad)
        assert grad_l[0] == '0.0'
        assert len(grad_l) == 8
        if temp == '':
            temp = '0'
        col_fl = fl[0]

        column_params = [company_name, usp_code, str(col_length), str(col_innerdiam), str(col_part_size), temp, col_fl, col_dead, HPLC_type, A_solv.split('.')[-1], B_solv.split('.')[-1]]
        column_params.extend(grad_l)

        if len(A_pH) > 1:
            A_pH = A_pH[1]
        if len(B_pH) > 1:
            B_pH = B_pH[1]
        column_params.extend([A_pH, B_pH])

        while len(A_start) > 1:
            A_start = A_start[1]
        while len(B_end) > 1:
            B_end = B_end[1]
        while len(A_end) > 1:
            A_end = A_end[1]
        while len(B_start) > 1:
            B_start = B_start[1]

        #print(tanaka_dict)
        try:
            tanaka_params = tanaka_dict[column_name]
        except:
            tanaka_params = np.zeros((8))
        try:
            
            hsmb_params = hsmb_dict[column_name]
        except:
            hsmb_params = np.zeros((7))
            
        column_params.extend([A_start, A_end, B_start, B_end])
        column_params.extend(A_add[1])

        column_params.extend(B_add[1])
        column_params.extend(tanaka_params)
        column_params.extend(hsmb_params)
        assert len(column_params) ==  len(params_header)
        #print(len(column_params))
        #exit()
        col_dict[dir] = column_params

        pre_data = [dir]
        pre_data.extend(grad_l)

    # Pickle the column metadata dictionary
    print(n_rm, 'REMOVED DUE TO RT FILTERING FOR NOT RETAINED') # can just keep this incase we get the filter list

    # modify pickle output name
    with open(f'{output}/{output_file}', 'wb') as file:
        pickle.dump(col_dict, file)
        print('pickled')
        break
    file.close()

# Comment out for now. might be necessary or not later 
# unique_values, counts = np.unique(np.asarray(removedRT), return_counts=True)
# print(unique_values, counts)
# for value, count in zip(unique_values, counts):
#     print(f"{value}: {count}")

print("EOF")
