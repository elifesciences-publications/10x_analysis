import pickle
import re
import os
from os.path import isfile, join
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import statistics
import math

class Track:
    def __init__(self, ID, times, positions):
        self.ID = ID
        self.times = times  # tuple of times
        self.positions = positions  # tuple of positions (positions are also tuple (x, y) )

# 211119
# path to folder of stitched ims
w572_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/w572_stitched'
w470_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/w470_stitched'
main_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_'
# path to tracks.pickle
df_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/track_df_w_CryS.pickle'


# 291119
# w572_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/w572_stitched'
# w470_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/w470_stitched'
# main_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/'
# df_path =  '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/track_df_w_CryS.pickle'

# pickle in background dict
pickle_in = open(main_dir + '/background_dict', 'rb')
background_dict = pickle.load(pickle_in)
pickle_in.close()

dict_keys = list(background_dict.keys())
dict_vals = list(background_dict.values())

# to contain datatime objects
deltaT_list = [None]*len(dict_keys)

for i in range(0, len(dict_keys)):
    deltaT_list[i] = int(re.findall('(?<=deltaT)(.*)(?=secs_)', dict_keys[i])[0])
print(deltaT_list)


# sort flist based on deltaT list
deltaT_list, dict_vals = zip(*sorted(zip(deltaT_list, dict_vals)))  # OMG, it works!
print(dict_keys)




# pickle in tracks
pickle_in = open(df_path, 'rb')
df = pickle.load(pickle_in)
pickle_in.close()


# get rid of short tracks
min_len = 4
df = df.loc[df['positions'].apply(len) >= min_len]




# substract background from CryS measurements
def subtract_background(CryS, frames, background):
    new_CryS = CryS
    for i in range(0, len(frames)):
        frame = int(frames[i])
        if frame < len(background):
            new_CryS[i] = CryS[i] - background[frame]
    return new_CryS

df['CryS'] = df.apply(lambda row: subtract_background(row['CryS'], row['times'], dict_vals), axis=1)




# convert positions from pixels to um
pixel_size = 1.098901  # 10x bottom Cairn

def pixels_to_um(positions, pixel_size):
    new_positions = list(positions)
    for i in range(0, len(positions)):
        new_positions[i] = (positions[i][0] * pixel_size, positions[i][1] * pixel_size)

    return tuple(new_positions)

df['positions'] = df.apply(lambda row: pixels_to_um(row['positions'], pixel_size), axis=1)


# convert times from frame to seconds
frame_interval = 2 * 60  # seconds

def frames_to_seconds(times, frame_interval):
    new_times = list(times)
    for i in range(0, len(times)):
        new_times[i] = times[i] * frame_interval
    return tuple(new_times)

df['times'] = df.apply(lambda row: frames_to_seconds(row['times'], frame_interval), axis=1)




# calculate speed
from common_track_measures import common_track_measures
df = common_track_measures(df)



## plot mean speed over time

# get max track len
max_len = 0
times = 0
for index, row in df.iterrows():
    if len(row['speeds']) > max_len:
        max_len = len(row['speeds'])
        times = row['times'][0:-1]

speeds = np.zeros((len(df), max_len))
speeds = np.full_like(speeds, np.nan)

for i in range(0, len(df)):
    row = df.iloc[i]
    speeds[i, 0:len(row['speeds'])] = row['speeds']

mean_speeds = np.nanmean(speeds, axis=0)

times = [x / 3600 for x in times]

plt.plot(times, mean_speeds)
plt.ylabel('Speed (um/s)')
plt.xlabel('Time (hrs)')
plt.title('Mean speed')
plt.savefig(main_dir + '/mean_speed.png', format='png')
plt.savefig(main_dir + '/mean_speed.svg', format='svg')
plt.show()
plt.close()



## plot mean CryS over time

# get max track len
max_len = 0
times = 0
for index, row in df.iterrows():
    if len(row['CryS']) > max_len:
        max_len = len(row['CryS'])
        times = row['times']

cryS = np.zeros((len(df), max_len))
cryS = np.full_like(cryS, np.nan)

for i in range(0, len(df)):
    row = df.iloc[i]
    cryS[i, 0:len(row['CryS'])] = row['CryS']

mean_CryS = np.nanmean(cryS, axis=0)

times = [x / 3600 for x in times]

plt.plot(times, mean_CryS)
plt.ylabel('CryS level (AU)')
plt.xlabel('Time (hrs)')
plt.title('Mean CryS')
plt.savefig(main_dir + '/mean_CryS.png', format='png')
plt.savefig(main_dir + '/mean_CryS.svg', format='svg')
plt.show()
plt.close()




# make col of initial CryS level

def mean_cryS(cryS, start_ind, stop_ind):

    return statistics.mean(cryS[start_ind: stop_ind])

df['CryS mean'] = df.apply(lambda row: mean_cryS(row['CryS'], start_ind=0, stop_ind=5), axis=1)


# create color map of initial CryS level
variable = 'CryS mean'

# get min and max of variable
max_val = max(df[variable])
min_val = min(df[variable])

norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
cmap = cm.viridis

m = cm.ScalarMappable(norm=norm, cmap=cmap)

df['color'] = df[variable].apply(m.to_rgba)


# plot speed and color by CryS
plt.figure(figsize=(10, 6))
for index, row in df.iterrows():
    speed = row['speeds']
    times = row['times'][0:-1]
    plt.plot(times, speed, c=row['color'])

plt.title('Tracks coloured by intitial CryS')
plt.ylabel('Speed (um/s)')
plt.xlabel('Time (s)')
plt.show()
plt.close()


# plot scatter of initial CryS level vs mean speed during time window

t_start_list = list(range(0, 100000, 7200))  # seconds
t_end_list = [x + (3600*3) for x in t_start_list] # seconds


fig, axs = plt.subplots(int(math.ceil(len(t_start_list)**0.5)), int(math.ceil(len(t_start_list)**0.5)),
                        figsize=(10, 10),
                        sharex=True,
                        sharey=True)

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Initial CryS (AU)")
plt.ylabel("Mean speed (um/s)")

matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)

for i in range(len(t_start_list)):

    print('working on subplot', i)

    t_start = t_start_list[i]
    t_end = t_end_list[i]

    axs = axs.flatten()
    ax = axs[i]

    n = 0

    for index, row in df.iterrows():

        times = row['times']
        indices = [idx for idx, val in enumerate(times) if (val >= t_start and val < t_end)]
        indices = indices[0: -1]
        if indices != []:
            speed_list = []
            for ind in indices:
                speed_list = speed_list + [row['speeds'][ind]]

            #print(speed_list)
            if len(speed_list) == 1:
                mean_speed = speed_list[0]
            else:
                mean_speed = statistics.mean(speed_list)
            ax.scatter(row['CryS mean'], mean_speed, c='blue', alpha=0.3, s=2)
            n += 1


    ax.set_title(str(t_start/3600) + ' to ' + str(t_end/3600) + ' hrs. (n = ' + str(n) + ')', size=9)

fig.suptitle('Initial CryS vs mean speed during different time windows')

plt.show()
plt.savefig(main_dir + '/initial_CryS_vs_speed.png', format='png')
plt.savefig(main_dir + '/initial_CryS_vs_speed.svg', format='svg')
plt.close()

#### 0 - 4hrs window

t_start = 0
t_end = 3600 * 4
n = 0

cryS_list = []
mean_speed_list = []

for index, row in df.iterrows():

    times = row['times']
    indices = [idx for idx, val in enumerate(times) if (val >= t_start and val < t_end)]
    indices = indices[0: -1]
    if indices != []:
        speed_list = []
        for ind in indices:
            speed_list = speed_list + [row['speeds'][ind]]

        # print(speed_list)
        if len(speed_list) == 1:
            mean_speed = speed_list[0]
        else:
            mean_speed = statistics.mean(speed_list)
        plt.scatter(row['CryS mean'], mean_speed, c='blue', alpha=0.3, s=2)
        n += 1
        cryS_list = cryS_list + [row['CryS mean']]
        mean_speed_list = mean_speed_list + [mean_speed]

percentile = np.percentile(np.array(cryS_list), 80)


## calculate Pearson's R

from scipy.stats import pearsonr

r, p = pearsonr(cryS_list, mean_speed_list)

print('Pearsons R = ', r, '. p = ', p)

# make df of data so John can plot in Prism
output = pd.DataFrame(list(zip(cryS_list, mean_speed_list)), columns =['Initial CryS level', 'Mean speed'])
output.to_csv(main_dir + '/initial_CryS_vs_speed_0_to_4hrs.csv')

plt.title('Intitial CryS level vs mean speed from 0-4hrs. (n = ' + str(n) + ').\n' + 'R = ' + str(r) + ', p = ' + str(p) + ' (Pearson)')
plt.xlabel('Initial CryS level (AU)')
plt.ylabel('Mean speed (um/s)')
plt.axvline(percentile, c='orange')

plt.show()
plt.savefig(main_dir + '/initial_CryS_vs_speed_0_to_4hrs.png', format='png')
plt.savefig(main_dir + '/initial_CryS_vs_speed_0_to_4hrs.svg', format='svg')
plt.close()

