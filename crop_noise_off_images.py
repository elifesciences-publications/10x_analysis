
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
data_dir = filedialog.askdirectory(parent=root, initialdir="/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x", title='Please select directory containing all directories with unstitched images')



# get list of all dirs in data dir, which containg the images to be stitched
image_dirs = next(os.walk(data_dir))[1]

for dir in image_dirs:

    # get list of images fnames
    flist = [f for f in os.listdir(data_dir + '/' + dir) if isfile(join(data_dir + '/' + dir, f))]
    print(flist)

    # remove any file containg 'stitched' from flist
    flist = [x for x in flist if 'stitched' not in x]

    for f in flist:
        image = cv.imread(data_dir + '/' + dir +'/' + f, -1)  # using cv2 https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images
        image = image[8:-10, 2:-2]  # crop out noise
        image = PIL.Image.fromarray(image)  # why is this not working now!?
        image.save(data_dir + '/' + dir +'/' + f)


