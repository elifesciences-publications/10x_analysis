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
from scipy import stats
from scipy.stats import linregress


# 211119
# # path to folder of stitched ims
# w572_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/w572_stitched'
# w470_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/w470_stitched'
# main_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_'
# # path to tracks.pickle
# df_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/track_df_w_CryS.pickle'
# manual_data_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/tracking_divs/Results_w_extra_measures.csv'
# save_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/'

# 291119
w572_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/w572_stitched'
w470_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/w470_stitched'
main_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/'
df_path =  '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/track_df_w_CryS.pickle'
manual_data_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/track_divs/Results_w_extra_measures.csv'
save_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/'
first_image_path = w572_dir + '/deltaT0secs_w572_proj_Fri Nov 29 0827h22s529 2019_stitched.tif'


# pickle in tracks_df
pickle_in = open(df_path, 'rb')
auto_df = pickle.load(pickle_in)
pickle_in.close()

# read in manual data
manual_df = data = pd.read_csv(manual_data_path)

# rotate manual data 90 CCW

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotate_point_90_CCW(x, y):
    x, y = rotate((0, 0), (x, y), math.radians(-90))

    return x, y

for i in range(0, len(manual_df)):
    # manual_df['Nuclear x'].iloc[i], manual_df['Nuclear y'].iloc[i] = rotate_point_90_CCW(manual_df['Nuclear x'].iloc[i], manual_df['Nuclear y'].iloc[i])
    manual_df.at[i, 'Nuclear x'], manual_df.at[i, 'Nuclear y'] = rotate_point_90_CCW(manual_df['Nuclear x'].iloc[i],
                                                                                         manual_df['Nuclear y'].iloc[i])


# offset manual data to match auto data
for i in range(0, len(manual_df)):
    # manual_df['Nuclear x'].iloc[i] = manual_df['Nuclear x'].iloc[i] + 0 - 9.79
    # manual_df['Nuclear y'].iloc[i] = manual_df['Nuclear y'].iloc[i] + 3400 - 22.21
    manual_df.at[i, 'Nuclear x'] = manual_df['Nuclear x'].iloc[i] + 0 - 9.79
    manual_df.at[i, 'Nuclear y'] = manual_df['Nuclear y'].iloc[i] + 3400 - 22.21

# swap x and y
for i in range(0, len(manual_df)):
    y = manual_df['Nuclear y'].iloc[i]
    x = manual_df['Nuclear x'].iloc[i]

    manual_df.at[i, 'Nuclear x'] = y
    manual_df.at[i, 'Nuclear y'] = x





# plot starting positions for auto data
for index, row in auto_df.iterrows():
    plt.scatter(row['positions'][0][0], row['positions'][0][1], color='k', s=1, alpha=0.5)

plt.axis((0, 3500, 0, 3500))
plt.gca().set_aspect('equal')
#plt.savefig(save_dir + 'auto.svg', format='svg')
#plt.show()
#plt.close()

# plot starting positions for manual data
for index, row in manual_df.iterrows():
    plt.scatter(row['Nuclear x'], row['Nuclear y'], color='r', s=1, alpha=0.5)
plt.axis((0, 3500, 0, 3500))
plt.gca().set_aspect('equal')
#plt.savefig(save_dir + 'manual.svg', format='svg')
plt.show()
#plt.close()




# get auto IDs for each row of manual_df

def get_displacement(pos0, pos1):
    '''
    Return the displacement between to coordinated in the form (x, y)
    :param pos0:
    :param pos1:
    :return:
    '''

    displacement = ((pos0[0] - pos1[0])**2 + (pos0[1] - pos1[1])**2)**0.5

    return displacement

manual_df['ID'] = 'None'
r = 10

print('getting IDs')
for i in range(0, len(manual_df)):
    print(i)
    ID_list = []
    man_x = manual_df['Nuclear x'].iloc[i]
    man_y = manual_df['Nuclear y'].iloc[i]

    for j in range(0, len(auto_df)):
        auto_x = auto_df['positions'].iloc[j][0][0]
        auto_y = auto_df['positions'].iloc[j][0][1]
        #print(auto_x, auto_y)

        distance = get_displacement((man_x, man_y), (auto_x, auto_y))
        if distance <= r:
            print('d=', distance)
            ID_list = ID_list + [auto_df['ID'].iloc[j]]

    if len(ID_list) == 1:
        manual_df['ID'].iloc[i] = ID_list[0]
    elif len(ID_list) < 1:
        print('No ID found')
    elif len(ID_list) > 1:
        print('More than one ID found')



# add div frame to auto_df

auto_df['div frame'] = np.nan

for i in range(0, len(auto_df)):
    auto_ID = auto_df['ID'].iloc[i]

    temp_manual_df = manual_df[manual_df['ID'] == auto_ID]
    if len(temp_manual_df) == 1:
        div_frame = temp_manual_df['Div frame'].values[0]
        auto_df['div frame'].iloc[i] = div_frame



## add track measures and convert units

# convert positions from pixels to um
pixel_size = 1.098901  # 10x bottom Cairn

def pixels_to_um(positions, pixel_size):
    new_positions = list(positions)
    for i in range(0, len(positions)):
        new_positions[i] = (positions[i][0] * pixel_size, positions[i][1] * pixel_size)

    return tuple(new_positions)

auto_df['positions'] = auto_df.apply(lambda row: pixels_to_um(row['positions'], pixel_size), axis=1)


# convert times from frame to seconds
frame_interval = 2 * 60  # seconds

def frames_to_seconds(times, frame_interval):
    new_times = list(times)
    for i in range(0, len(times)):
        new_times[i] = times[i] * frame_interval
    return tuple(new_times)

auto_df['times'] = auto_df.apply(lambda row: frames_to_seconds(row['times'], frame_interval), axis=1)


# get rid of short tracks
min_len = 4
auto_df = auto_df.loc[auto_df['positions'].apply(len) >= min_len]



# calculate speed
from common_track_measures import common_track_measures
auto_df = common_track_measures(auto_df)






auto_df_w_manual_data = auto_df[auto_df['div frame'].isnull() == False]
auto_df_w_manual_data = auto_df[auto_df['div frame'] != 1]
auto_df_w_manual_data_non_dividers = auto_df[auto_df['div frame'] == 1]


for index, row in auto_df_w_manual_data.iterrows():
    if row['div frame'] > 1:
        plt.scatter(row['div frame'], row['median speed'])

plt.show()

### hist

t, p = stats.ks_2samp(auto_df_w_manual_data['average speed'].values, auto_df_w_manual_data_non_dividers['average speed'].values)
thresh = 0.05
sig = 'NS'
if p < thresh:
    sig = 'Significant'

print('Kolmogorov-Smirnov')
print('p=', p, sig)

n_dividers = len(auto_df_w_manual_data)
n_non_dividers = len(auto_df_w_manual_data_non_dividers)

divider_label = 'Dividers (n=' + str(n_dividers) + ')'
non_divider_label = 'Non-dividers (n=' + str(n_non_dividers) + ')'

plt.hist(auto_df_w_manual_data['average speed'].values, alpha=1, label=divider_label, color='blue', histtype=u'step')
plt.hist(auto_df_w_manual_data_non_dividers['average speed'].values, alpha=1, label=non_divider_label, color='green', histtype=u'step')
plt.xlabel('Average speed')
plt.ylabel('Frequency (%)')
plt.legend(loc='upper right')
plt.title('p=' + str(p))
# plt.savefig(save_path + '/histogram.png', format='png')
# plt.savefig(save_path + '/histogram.svg', format='svg')
plt.show()





# plot scatter of initial CryS level vs mean speed during time window

t_start_list = list(range(0, 100000, 3600))  # seconds
t_end_list = [x + (3600*1) for x in t_start_list] # seconds


fig, axs = plt.subplots(int(math.ceil(len(t_start_list)**0.5)), int(math.ceil(len(t_start_list)**0.5)),
                        figsize=(10, 10),
                        sharex=True,
                        sharey=True)

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Division time")
plt.ylabel("Mean speed (um/s)")

matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)

for i in range(len(t_start_list)):

    print('working on subplot', i)

    t_start = t_start_list[i]
    t_end = t_end_list[i]

    axs = axs.flatten()
    ax = axs[i]

    for index, row in auto_df_w_manual_data.iterrows():

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
            ax.scatter(row['div frame'], mean_speed, c='blue', alpha=0.3, s=2)


    ax.set_title(str(t_start/3600) + ' to ' + str(t_end/3600) + ' hrs', size=10)

fig.suptitle('Division time vs mean speed during different time windows')

plt.show()
plt.savefig(main_dir + '/division_time_vs_speed.png', format='png')
plt.savefig(main_dir + '/division_time_vs_speed.svg', format='svg')
plt.close()




##### clustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

df = auto_df[auto_df['div frame'].isnull() == False]
df.replace(1, np.nan, inplace=True)

# get rid of short tracks
min_len = 150
df = df.loc[df['positions'].apply(len) >= min_len]



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

X = speeds
X = X[:, 0:min_len]

# create dendro
#dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

# cluster
model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
model.fit(X)
labels = model.labels_

cluster_1 = X[(labels == 0), :]
cluster_2 = X[(labels == 1), :]

cluster_1_mean = np.mean(cluster_1, axis=0)
cluster_2_mean = np.mean(cluster_2, axis=0)


plt.plot(cluster_1_mean, c='red', label='Cluster 1', alpha=0.5)
plt.plot(cluster_2_mean, c='blue', label='Cluster 2', alpha=0.5)
plt.legend(loc='upper right')
plt.show()


# add labels to df
df['cluster'] = labels



variable = 'cluster'

# get min and max of variable
max_val = max(df[variable])
min_val = min(df[variable])

norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
cmap = cm.viridis

m = cm.ScalarMappable(norm=norm, cmap=cmap)

df['color'] = df[variable].apply(m.to_rgba)

plt.figure(figsize=(10, 6))
for index, row in df.iterrows():
    plt.plot(row['times'][0:-1], row['speeds'], c=row['color'], alpha=0.5)

plt.title('Tracks coloured by ' + variable)
plt.show()



### cluster hist

df_non_div = df[df['div frame'].isnull()]
df_div = df[df['div frame'].isnull() == False]

df_cluster_1 = df[df['cluster'] == 0]
df_cluster_2 = df[df['cluster'] == 1]

df_div_cluster_1 = df_div[df_div['cluster'] == 0]
df_div_cluster_2 = df_div[df_div['cluster'] == 1]


t, p = stats.ks_2samp(df_div_cluster_1['div frame'].values, df_div_cluster_2['div frame'].values)
thresh = 0.05
sig = 'NS'
if p < thresh:
    sig = 'Significant'

print('Kolmogorov-Smirnov')
print('p=', p, sig)

n_cluster_1 = len(df_div_cluster_1)
n_cluster_2 = len(df_div_cluster_2)

cluster_1_label = 'Cluster 1 (n=' + str(n_cluster_1) + ')'
cluster_2_label = 'Cluster 2 (n=' + str(n_cluster_2) + ')'

plt.hist(df_div_cluster_1['div frame'].values, alpha=1, label=cluster_1_label, color='blue', histtype=u'step')
plt.hist(df_div_cluster_2['div frame'].values, alpha=1, label=cluster_2_label, color='green', histtype=u'step')
plt.xlabel('Division frame')
plt.ylabel('Frequency (%)')
plt.legend(loc='upper right')
plt.title('p=' + str(p))
# plt.savefig(save_path + '/histogram.png', format='png')
# plt.savefig(save_path + '/histogram.svg', format='svg')
plt.show()


pct_dividers = (len(df[df['div frame'].isnull() == False]) / len(df)) * 100

pct_dividers_cluster_1 = (len(df_cluster_1[df_cluster_1['div frame'].isnull() == False]) / len(df_cluster_1)) * 100

pct_dividers_cluster_2 = (len(df_cluster_2[df_cluster_2['div frame'].isnull() == False]) / len(df_cluster_2)) * 100

print('pct dividers cluster 1 and 2', pct_dividers)
print('pct dividers cluster 1', pct_dividers_cluster_1)
print('pct dividers cluster 2', pct_dividers_cluster_2)

#### test sig ####
import random

repeats = 5000
pct_dividers_cluster_2_sig_tes_list = [None] * repeats

for i in range(0, repeats):
    print(i)

    # assign cluster randomly, with same number as real data
    df['cluster'] = 0
    # assign cluster 2
    df['cluster'].iloc[0:n_cluster_2] = 1
    # shuffle cluster
    clusters = list(df['cluster'].values)
    random.shuffle(clusters)
    df['cluster'] = clusters

    # calc pct dividers in each cluster
    df_cluster_1 = df[df['cluster'] == 0]
    df_cluster_2 = df[df['cluster'] == 1]

    pct_dividers_sig_test = (len(df[df['div frame'].isnull() == False]) / len(df)) * 100
    pct_dividers_cluster_1_sig_test = (len(df_cluster_1[df_cluster_1['div frame'].isnull() == False]) / len(df_cluster_1)) * 100
    pct_dividers_cluster_2_sig_test = (len(df_cluster_2[df_cluster_2['div frame'].isnull() == False]) / len(df_cluster_2)) * 100

    pct_dividers_cluster_2_sig_tes_list[i] = pct_dividers_cluster_2_sig_test


more_biased_list = [x for x in pct_dividers_cluster_2_sig_tes_list if x <= pct_dividers_cluster_2]

pct_by_chance = (len(more_biased_list) / repeats) * 100
print('pct by chance', pct_by_chance)