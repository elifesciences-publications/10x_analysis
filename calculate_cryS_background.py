import pickle
import re
import os
from os.path import isfile, join
import cv2 as cv
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
import pandas as pd

class Track:
    def __init__(self, ID, times, positions):
        self.ID = ID
        self.times = times  # tuple of times
        self.positions = positions  # tuple of positions (positions are also tuple (x, y) )


# 211119
# path to folder of stitched ims
# w572_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/w572_stitched'
# w470_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/w470_stitched'
# main_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/'
# # features path
# features_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/features_df.pickle'
# # path to tracks.pickle
# tracks_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/track_list.pickle'


# 291119
# path to folder of stitched ims
w572_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/w572_stitched'
w470_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/w470_stitched'
main_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/'
# features path
features_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/features_df.pickle'
# path to tracks.pickle
tracks_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/track_list.pickle'


# pickle in tracks
pickle_in = open(tracks_path, 'rb')
track_list = pickle.load(pickle_in)
pickle_in.close()


# pickle in features
pickle_in = open(features_path, 'rb')
features_df = pickle.load(pickle_in)
pickle_in.close()


## create ordered list of stitched images

w572_flist = [f for f in os.listdir(w572_dir) if isfile(join(w572_dir, f))]

# include only files which include 'w572_top'
w572_flist = [f for f in w572_flist if 'w572_' in f]

print(w572_flist)

# to contain datatime objects
deltaT_list = [None]*len(w572_flist)

for i in range(0, len(w572_flist)):
    deltaT_list[i] = int(re.findall('(?<=deltaT)(.*)(?=secs_)', w572_flist[i])[0])
print(deltaT_list)


# sort flist based on deltaT list
deltaT_list, w572_flist = zip(*sorted(zip(deltaT_list, w572_flist)))  # OMG, it works!
print(w572_flist)

## w470

w470_flist = [f for f in os.listdir(w470_dir) if isfile(join(w470_dir, f))]

# include only files which include 'w572_top'
w470_flist = [f for f in w470_flist if 'w470_' in f]

print(w470_flist)

# to contain datatime objects
deltaT_list = [None]*len(w470_flist)

for i in range(0, len(w470_flist)):
    deltaT_list[i] = int(re.findall('(?<=deltaT)(.*)(?=secs_)', w470_flist[i])[0])
print(deltaT_list)


# sort flist based on deltaT list
deltaT_list, w470_flist = zip(*sorted(zip(deltaT_list, w470_flist)))  # OMG, it works!
print(w470_flist)


# correct for y axis offset compared to ims
y_offset = 8


## test plotting circles round points

frame = 0

dir = w572_dir
im_path = w572_flist[frame]

image = cv.imread(dir + '/' + im_path, -1)  # using cv2 https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images

image = np.flip(image, axis=0)
image = np.rot90(image, 2)

frame_features = features_df[features_df['frame'] == frame]

plt.figure(figsize=(6, 6), dpi=300)
plt.imshow(image, vmin=103.8, vmax=121.6)

for index, row in frame_features.iterrows():
    plt.plot(row['x'], row['y'] + y_offset,
             marker='o',
             fillstyle='none',
             c='white',
             markersize=2,
             markeredgewidth=0.2)

plt.show()



## get background for each CryS frame, in order of flist
cryS_background = [None] * len(w470_flist)

# radius around features to set to NaN (pixels)
background_r = 20

for i in range(0, len(w470_flist)):
    print('working on background frame', i+1, 'of', len(w470_flist))
    frame = i

    dir = w470_dir
    im_path = w470_flist[frame]

    background_r = 20


    image = cv.imread(dir + '/' + im_path, -1)  # using cv2 https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images

    image = np.flip(image, axis=0)
    image = np.rot90(image, 2)

    frame_features = features_df[features_df['frame'] == frame]

    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])

    for index, row in frame_features.iterrows():
        cx = round(row['x'])
        cy = round(row['y'] + y_offset)

        circle = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < background_r ** 2
        image[circle] = np.nan

    cryS_background[i] = np.nanmedian(image)

    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(image, vmin=139.7, vmax=186.9)
    plt.show()


background_dict = dict(zip(w470_flist, cryS_background))

# pickle out background dict
pickle_out = open(main_dir + '/background_dict', 'wb')
pickle.dump(background_dict, pickle_out)
pickle_out.close()



