'''
Converts a list of Track objects to a pandas dataframe, where each row is a track.
'''

import pickle
import pandas as pd
import statistics

def mean_of_flags(flags):
    flags = list(flags)
    return statistics.mean(flags)

def track_list_to_df(track_list, is_model=True):

    ## if data is from C++ model (default)
    if is_model == True:
        # create new df using col names, with no rows
        cols = ['ID', 'times', 'positions', 'folate chemo flags', 'cAMP stim flags', 'starved flags']
        df = pd.DataFrame(columns=cols)

        # iterate through track_list, adding attributes to new row of df
        for track in track_list:
            data = [[track.ID, track.times, track.positions, track.folate_chemo_flag, track.cAMP_stim_flag, track.starved_flag]]  # get data into list form
            temp_df = pd.DataFrame(data, columns=cols)  # create one-row df
            df = df.append(temp_df)  # append to main df

        # create cols for mean flags
        df['mean folate flag'] = df['folate chemo flags'].apply(mean_of_flags)
        df['mean cAMP flag'] = df['cAMP stim flags'].apply(mean_of_flags)
        df['mean starved flag'] = df['starved flags'].apply(mean_of_flags)

        return df

    ## if data is not from model, i.e. is from experimental images
    else:
        # create new df using col names, with no rows
        cols = ['ID', 'times', 'positions']
        df = pd.DataFrame(columns=cols)

        # iterate through track_list, adding attributes to new row of df
        counter = 1
        for track in track_list:
            print('converting track', counter, 'of', len(track_list))
            counter += 1
            data = [[track.ID, track.times, track.positions]]  # get data into list form
            temp_df = pd.DataFrame(data, columns=cols)  # create one-row df
            df = df.append(temp_df)  # append to main df

        return df




## testing

# path = '/media/jacob/data/Chubb_lab/notebook/feeding_front_decision/feeding_model/cpp/feeding_model_cpp/data/results/track_list.pickle'
#
# pickle_in = open(path, 'rb')
# track_list = pickle.load(pickle_in)
# pickle_in.close()
#
# df = track_list_to_df(track_list)
# print(df)