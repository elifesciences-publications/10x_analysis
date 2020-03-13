'''
Script to use trackpy to locate and track nuclei in unstitched collections of overlapping images.
'''

import trackpy as tp
import pims
import image_analysis_module2 as ia
import matplotlib.pyplot as plt
import os
from os.path import isfile, join, isdir
import re
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

import funcs


# class particle:
#     def __init__(self, track_df):
#         self.track_df = track_df
#
#         self.track_x_list = self.track_df['x'].tolist()  # x coordinated as list
#         self.track_y_list = self.track_df['y'].tolist()  # y coordinates as list
#         self.track_length_frames = self.track_df.shape[1]  # number of frames in track
#         self.track_length_pixels = track_length(self.track_x_list, self.track_y_list)
#         self.mean_speed = self.track_length_pixels/self.track_length_frames


plt.ion()

# 211119
# image_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/w572_proj_Fri Nov 22 0510h54s368 2019/572_bottom_X-32101,4_Y2604,45_Fri Nov 22 0511h13s753 2019.tif'
# timecourse_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_'

# 291119
image_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/w572_proj_Fri Nov 29 1151h05s090 2019/572_bottom_X-30772,6_Y-6564,16_Fri Nov 29 1151h13s065 2019.tif'
timecourse_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_'


stack = image_path
particle_size = 13
minmass = 145
pixel_size = 1.098901 # 10x bottom Cairn

# for linking
memory = 1  # number of frames a particle can disappear for
max_dislplacement = 50  # pixels


#funcs.locate_feature_calibrate(image_path, particle_size, minmass)


## get list of image-containg dirs in timecourse dir

dir_list = [f for f in os.listdir(timecourse_path) if isdir(join(timecourse_path, f))]
print(dir_list)

# get only those dirs containg 572_top
dir_list = [d for d in dir_list if 'w572_proj' in d]
print(dir_list)


## generate deltaT list in same order as dir_list

# to contain datatime objects
date_list = []

for d in dir_list:
    print(d)
    split = d.split(' ')

    print(split)

    year = split[-1]
    month = split[1]
    day = split[2]
    if len(day) == 1:
        day = '0' + day

    time = split[3]

    hour_min = time.split('h')[0]

    hour = hour_min[0:2]
    mins = hour_min[2:]

    seconds = re.findall('(?<=h)(.*)(?=s)',time)[0]

    datetime_str = year + ' ' + month + ' ' + day + ' ' + hour + ' ' + mins + ' ' + seconds
    print(datetime_str)

    datetime = datetime.strptime(datetime_str, '%Y %b %d %H %M %S')
    date_list = date_list + [datetime]


# for each element in date_list, calculate difference in seconds from fisrt
s_diff_list = [None]*len(date_list)
s_diff_list[0] = 0
for i in range(1, len(date_list)):
    s_diff_list[i] = (date_list[i] - date_list[0]).total_seconds()


# now subtradt min value from s_diff_list to get seconds since first image for each fname
s_diff_list = np.asarray(s_diff_list)
s_diff_list = s_diff_list - np.amin(s_diff_list)
s_diff_list = np.ndarray.tolist(s_diff_list)
s_diff_list = [round(x) for x in s_diff_list]
print(s_diff_list)


# sort dir_list based on s_diff_list
s_diff_list, dir_list = zip(*sorted(zip(s_diff_list, dir_list)))  # OMG, it works!
print(dir_list)



#dir_list = dir_list[-3:]
#s_diff_list = s_diff_list[-3:]


## locate features at each time point across all unstitched images

for i in range(0, len(dir_list)):

    print('Working on image', i+1, 'of', len(dir_list))
    dir_path = timecourse_path + '/' + dir_list[i]

    time = s_diff_list[i]/3600

    # locate features
    f = funcs.locate_across_image_collection(dir_path, pixel_size, particle_size, minmass)

    # add a frame col
    f['frame'] = i

    if i == 0:
        f_total = f
    else:
        f_total = f_total.append(f, sort=True)


# pickle out features df
pickle_out = open(timecourse_path + '/features_df.pickle', 'wb')
pickle.dump(f_total, pickle_out)
pickle_out.close()

# pickle in features df
pickle_in = open(timecourse_path + '/features_df.pickle', 'rb')
f_total = pickle.load(pickle_in)
pickle_in.close()


# plot num features vs frame
frames = list(set(list(f_total['frame'].values)))
num_particles = []
for frame in frames:
    temp = f_total[f_total.frame == frame]
    num_particles = num_particles + [len(temp)]

plt.plot(frames, num_particles)
plt.xlabel('Frame')
plt.ylabel('Number of features detected')
plt.show()


# link features into tracks
t = tp.link_df(f_total,  memory=memory, adaptive_stop=2, search_range=max_dislplacement, adaptive_step=0.95)

##reformat output to get list of Tracks objects

class Track:
    def __init__(self, ID, times, positions):
        self.ID = ID
        self.times = times  # tuple of times
        self.positions = positions  # tuple of positions (positions are also tuple (x, y) )

tracks = []

# get list of all unique track IDs
track_list = t['particle'].tolist()
track_set = set(track_list)

for track in track_set:
    curr_track = t.loc[t['particle'] == track]

    # order curr_track df by frame
    curr_track = curr_track.sort_values('frame')

    # create times
    times = tuple(curr_track['frame'].values)

    # create positions
    x_list = list(curr_track['x'].values)
    y_list = list(curr_track['y'].values)

    positions = [None] * len(x_list)
    for i in range(0, len(x_list)):
        pos = (x_list[i], y_list[i])
        positions[i] = pos

    positions = tuple(positions)


    tracks = tracks + [Track(ID=track, times=times, positions=positions)]


## interpoalte missing frames

# If memory is set to greater than 0, then tracks may be made up of points at non-consecutive frames,
# i.e. point 0 might be frame 0, and point 1 frame 2. This causes problems in downstream analysis.
# In order to deal with this, missing frames in a track will have a position calculated based on linear interpolation.

from interpolate_between_points import interpolate_points_between_points_tracks
tracks = interpolate_points_between_points_tracks(tracks)


# pickle out track list
pickle_out = open(timecourse_path + '/track_list.pickle', 'wb')
pickle.dump(tracks, pickle_out)
pickle_out.close()
