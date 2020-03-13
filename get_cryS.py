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
# # path to tracks.pickle
# tracks_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/track_list.pickle'
# # features path
# features_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/features_df.pickle'

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



# pickle in background dict
# pickle_in = open(main_dir + '/background_dict', 'rb')
# background_dict = pickle.load(pickle_in)
# pickle_in.close()
#

r = 20


new_track_list = []
for track in track_list:
    if (track.times[0] == 0):
        new_track_list = new_track_list + [track]

track_list = new_track_list

## convert track_list to df
from track_list_to_df import track_list_to_df

print('creating df')
track_df = track_list_to_df(track_list, is_model=False)

track_df['CryS'] = None

#track_df = track_df.iloc[0:2]

print('measuring cryS')
for j in range(0, len(track_df)):
    print('getting CryS for track', j+1, 'of', len(track_df))

    row = track_df.iloc[j]
    frames = row['times']
    cryS = [None] * len(frames)
    x_list = [item[0] for item in row['positions']]
    y_list = [item[1] for item in row['positions']]

    for i in range(0, len(frames)):
        frame = int(frames[i])  # number frame of current pos


        if frame < len(w470_flist):
            x_pos = x_list[i]
            y_pos = y_list[i] + y_offset

            #background = background_dict[w470_flist[frame]]
            background = 0

            image = cv.imread(w470_dir + '/' + w470_flist[frame], -1)  # using cv2 https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images

            image = np.flip(image, axis=0)
            image = np.rot90(image, 2)

            x = np.arange(0, image.shape[1])
            y = np.arange(0, image.shape[0])

            cx = round(x_pos)
            cy = round(y_pos)

            circle = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2

            cryS_level = image[circle] - background
            cryS_level = np.nanmean(cryS_level)
            cryS[i] = cryS_level

    track_df['CryS'].iloc[j] = cryS

# pickle ot df
pickle_out = open(main_dir + '/track_df_w_CryS.pickle', 'wb')
pickle.dump(track_df, pickle_out)
pickle_out.close()