'''
Script to stitch together images in a dir.

Images should have been acquired on the Cairn by my journal(s) which saves each image tile in a directory for that
time point and channel, such that each dir contains only the images to be stitched.

Note: This project is using the system Python3 interpreter.

'''

import os
import image_analysis_module2 as IA  # my custom module
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tkinter import filedialog
from tkinter import *


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



# for each dir, stitch the containing images and save in containing dir, and dir containing all images for each channel

# create dirs to store all images in one place
w470_dir = data_dir + '/' + 'w470_stitched'
os.mkdir(w470_dir)
w572_dir = data_dir + '/' + 'w572_stitched'
os.mkdir(w572_dir)
wExt_dir = data_dir + '/' + 'wExt_stitched'
os.mkdir(wExt_dir)

dir_num = 0

for dir in image_dirs:

    dir_num += 1
    print("Stitching dir " + str(dir_num) + " out of " + str(len(image_dirs)))


    print(dir)
    pixel_size = 1.098901 # 10x bottom Cairn
    stitched_image = IA.xy_stitch_dir_of_images(data_dir + '/' + dir, pixel_size, 1)
    print(type(stitched_image))
    print('image stitched')
    #plt.imshow(stitched_image)
    #plt.show()

    os.chdir(data_dir + '/' + dir)
    stitched_image = PIL.Image.fromarray(stitched_image)  # why is this not working now!?
    im_name = dir + '_stitched.tif'
    stitched_image.save(im_name)

    if '470' in im_name:
        os.chdir(w470_dir)
        stitched_image.save(im_name)
    if '572' in im_name:
        os.chdir(w572_dir)
        stitched_image.save(im_name)
    if 'ext_top' in im_name:
        os.chdir(wExt_dir)
        stitched_image.save(im_name)

os.chdir(data_dir)








