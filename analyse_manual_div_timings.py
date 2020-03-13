import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress

# 211119
# csv_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/tracking_divs/Results_w_extra_measures.csv'
# save_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/tracking_divs'

# 291119
csv_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/track_divs/Results_w_extra_measures.csv'
save_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/track_divs'

data = pd.read_csv(csv_path)



# add new cols
data['mean CryS minus median background'] = data['mean CryS'] - data['median background']
data['mean CryS minus median background small area'] = data['mean CryS 12 pix radius'] - data['median background']


CryS_col = 'mean CryS minus median background small area'


dividers = data[data['Div frame'] != 1]

non_dividers = data[data['Div frame'] == 1]

n_dividers = len(dividers)
n_non_dividers = len(non_dividers)

divider_label = 'Dividers (n=' + str(n_dividers) + ')'
non_divider_label = 'Non-dividers (n=' + str(n_non_dividers) + ')'


t, p = stats.ttest_ind(dividers[CryS_col].values, non_dividers[CryS_col].values)
thresh = 0.05
sig = 'NS'
if p < thresh:
    sig = 'Significant'

print('t-test')
print('p=', p, sig)

t, p = stats.ks_2samp(dividers[CryS_col].values, non_dividers[CryS_col].values)
thresh = 0.05
sig = 'NS'
if p < thresh:
    sig = 'Significant'

print('Kolmogorov-Smirnov')
print('p=', p, sig)



plt.hist(dividers[CryS_col].values, alpha=1, label=divider_label, color='blue', histtype=u'step')
plt.hist(non_dividers[CryS_col].values, alpha=1, label=non_divider_label, color='green', histtype=u'step')
plt.xlabel('Mean CryS level (AU)')
plt.ylabel('Frequency (%)')
plt.legend(loc='upper right')
plt.title('p=' + str(p))
# plt.savefig(save_path + '/histogram.png', format='png')
# plt.savefig(save_path + '/histogram.svg', format='svg')
plt.show()





slope, intercept, r, p, stderr = linregress(dividers['Div frame'].values, dividers[CryS_col].values)
print(linregress(dividers['Div frame'].values, dividers[CryS_col].values))

div_time = [x/30 for x in dividers['Div frame'].values]

plt.scatter(div_time, dividers[CryS_col].values)
#plt.scatter(div_time, dividers['mean CryS'].values)
plt.xlabel('Division time (hrs)')
plt.ylabel('Mean CryS level (AU)')
plt.title('N dividers = ' + str(len(dividers)) + '\nr=' + str(r) + ', p=' + str(p))
# plt.savefig(save_path + '/CryS_vs_div_time.png', format='png')
# plt.savefig(save_path + '/CryS_vs_div_time.svg', format='svg')
plt.show()



plt.hist(dividers['Div frame'].values, alpha=1, label=divider_label, color='blue', histtype=u'step')
plt.show()