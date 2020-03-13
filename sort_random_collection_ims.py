
import os
from os.path import isfile, join
import image_analysis_module2 as IA  # my custom module
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tkinter import filedialog
from tkinter import *
import cv2 as cv


# set working dir as parent dir that contains all the dirs for different time points and channels
#working_dir = '/home/jacob/Documents/Chubb_lab/notebook/why_beome_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/analysis'
#os.chdir(working_dir)




## go through each dir and stitch images within based on x and y coordinates ##

# set dir that contains all the dirs for different time points and channels
#data_dir = '/home/jacob/Documents/Chubb_lab/notebook/why_beome_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/analysis/100419_JR3_testing'

# or let user select dir from dialog
root = Tk()
data_dir = filedialog.askdirectory(parent=root, initialdir="/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/131119_JR6_10x_bottom_testing", title='Please select directory containing all directories with unstitched images')

flist = [f for f in os.listdir(data_dir) if isfile(join(data_dir + '/', f))]
flist = [os.path.join(data_dir, f) for f in flist] # add path to each file

# sort flist based ion modification time
flist.sort(key=lambda x: os.path.getmtime(x))

# split into channels

# w470_list = []
#
# w572_list = []
#
# for f in flist:
#     if '572_bottom' in f:
#         w572_list = w572_list + [f]
#     elif '470_bottom' in f:
#         w470_list = w470_list + [f]



w572_frame_counter = 1
w470_frame_counter = 1
for i in range(0, len(flist), 9):

    collection = flist[i: i + 9]
    if '572_bottom' in collection[0]:
        new_dir = 'w572_' + str(w572_frame_counter)
        w572_frame_counter += 1
    else:
        new_dir = 'w470_' + str(w470_frame_counter)
        w470_frame_counter += 1

    os.mkdir(data_dir + '/' + new_dir)

    for f in collection:
        fname = os.path.basename(f)
        os.rename(f, data_dir + '/' + new_dir + '/' + fname)



