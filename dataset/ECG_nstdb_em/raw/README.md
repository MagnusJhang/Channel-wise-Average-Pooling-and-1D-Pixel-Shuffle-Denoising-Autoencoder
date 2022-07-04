# MIT-BIH Noise Street Test Database

*The dataset are obtains from PhysioNet(DOI: https://doi.org/10.13026/C2HS3T)*

We download 12 file(118e00, 118e06, 118e12, 118e18, 118e24, 118e_6, 119e00, 119e06, 119e12, 119e18, 119e24, and 119e_6)
and transfer these file into the .mat file, which means it also can be opened by matlab.

There are total number of 12 mat file in the folder, and the SNR during the noisy segments of these are:

| File Name        | SNR_in (dB) | File Name        | SNR_in (dB) |
|------------------|-------------|------------------|-------------|
| nstdb_118e00.mat | 0           | nstdb_119e00.mat | 0           |
| nstdb_118e06.mat | 6           | nstdb_119e06.mat | 6           |
| nstdb_118e12.mat | 12          | nstdb_119e12.mat | 12          |
| nstdb_118e18.mat | 18          | nstdb_119e18.mat | 18          |
| nstdb_118e24.mat | 24          | nstdb_119e24.mat | 24          |
| nstdb_118e-6.mat | -6          | nstdb_119e-6.mat | -6          |

The structure of the mata file is shown as following:
* Fs: sampling frequency (360 Hz)
* length: number of samples (30 min)
* vector: number of leads (2 leads)
* description: the leads information (eg.: MLII, Lead I, Lead II, etc.)
* binary: the raw data of noisy ECG
* binary_zb: the DC-free version of raw data of noisy ECG
* binary_clear: the corresponding clean ECG of noisy ECG
* binary_zb_clear: the DC-free version of the corresponding clean ECG

## How to use:
In matlab:
```matlab
load("./nstdb_118e00.mat");
length = 1024;
load_seg_st = 108000;
load_seg_ed = load_seg_st + length -1;
load_seg = load_seg_st:load_seg_ed;
load_lead = 1; 
noisy_ecg = data.binary_zb(load_seg, load_lead);
clean_ecg = data.binary_zb_clear(load_seg, load_lead);
figure();
hold on;
plot(load_seg, noisy_ecg, 'displayname', 'Noisy ECG');
plot(load_seg, clean_ecg, 'displayname', 'Clean ECG');
legend;
title(sprintf("nstdb 118e00[%d][%d-%d]", load_lead-1, load_seg_st-1, load_seg_ed-1));
%The start index is 1 in matlab%
```
In python3:
```python
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
d = sio.loadmat("./nstdb_118e00.mat")['data']
length = 1024
load_seg_st= 107999
load_seg_ed = load_seg_st+length
load_lead = 0
noisy_ecg = d['binary_zb'][0, 0][load_seg_st:load_seg_ed, load_lead]
clean_ecg = d['binary_zb_clear'][0, 0][load_seg_st:load_seg_ed, load_lead]
xaxis = np.linspace(load_seg_st, load_seg_ed-1, length)
plt.plot(xaxis, noisy_ecg, label='Noisy ECG')
plt.plot(xaxis, clean_ecg, label='Clean ECG')
plt.title("nstdb 118e00[{}][{}-{}]".format(load_lead, load_seg_st, load_seg_ed))
plt.legend(loc='best')
plt.show()
```