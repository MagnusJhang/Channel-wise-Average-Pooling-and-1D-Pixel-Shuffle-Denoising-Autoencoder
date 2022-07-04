#The project for 1D PixelShuffle Denoising Autoencoder with Channel-wise Average Pooling
<div align=right>Author: Yu-Syuan Jhang</div>

## Training and inference the CPDAE model
Most simply scription can type as following: 
```console
$ cd <path-of-the-project-ECG_Denoising-CPDAE>
./ECG_Denoising-CPDAE $ python3 main_ECG_NoiseDB_Mixed_v2.py CPDAE_Regular
``` 
The execution result will store in ./output/bs32_CPDAE_Regular. And the output structural is shown as below:

```console
ECG_Denoising-CPDAE $ tree ./output/bs_32_CPDAE_Regular 

bs32_CPDAE_Regular
├──dump
|   ├──epoch0001_train01.png
|    ...
|   ├──epoch0001_train10.png
|   ├──epoch0001_test01.png
|   ...
|   ├──epoch0001_test10.png
|               .
|               .
|               .
|   └──epoch0500_test10.png
├──test
|   ├──criteria_test.csv
|   ├──test_00000.png
|               .
|               .
|               .
|   └──test_01319.png (test_xxxxx.png is available only if '-df True' argument is set when excute main_CPDAE.py)
├──loss.csv
├──loss.png
├──model_state_dict.pt
└──summary.json
```
**dump folder**

To let the result visualized, 20 images (10 for training, 10 for testing) will be generated every 50 and last 10 epochs into ./output/<bsXX-model_name>/dump folder.

Each image has three series, Clean ECG, Corrupted ECG, and Reconstructed ECG.

---
**test folder**

* criteria_test.csv

	The evaluation criteria for test-dataset such as improvement of SNR, PRD, and RMSE are obtained at the final epoch.
	These criteria values are written into the criteria_test.csv. 
```console
./output/bs32_CPDAE_Regular $ cat ./test/ccriteria_test.csv
methods,fname,ch,idx,in_snr_dB,ou_snr_dB,SNR_imp,PRD,RMSE
CPDAE_Regular,nstdb_119e18,0.0,148960.0,18.0,15.874215126037598,12.258535385131836,16.08011817932129,0.0066211330704391
CPDAE_Regular,nstdb_119e18,0.0,128480.0,18.0,17.46068572998047,15.724570274353027,13.395711898803711,0.0044909026473760605
CPDAE_Regular,nstdb_119e18,0.0,632544.0,18.0,22.98996353149414,8.957569122314453,7.087641716003418,0.004213728941977024
CPDAE_Regular,nstdb_119e18,0.0,368224.0,18.0,22.74649429321289,8.07573127746582,7.289124488830566,0.004429951310157776
CPDAE_Regular,nstdb_119e18,0.0,370272.0,18.0,17.412132263183594,13.305695533752441,13.47079849243164,0.004893053788691759
                                                    .
                                                    .
                                                    .
CPDAE_Regular,nstdb_118e06,1.0,643808.0,6.0,8.029313087463379,20.187564849853516,39.67658996582031,0.013230782933533192
```
You can use any data analysis tool to obtain the distribution such as matlab, pandas, Microsoft Excel, IBM SPSS, etc.

* test_00000.png - test_0xxxx.png

	There is also provide a function to export all the image in inference stage in final epoch by adding the arguiment "**-df True**".


* loss.csv and loss.png

	The loss value of every epoch.
	
*  model_state_dict.pt

	The parameter of model will be exported at the final stage.

* summary.json

	The information of this experiment.
```console
./bs32_CPDAE_Regular $ cat ./summary
{
    "PRD": {
        "-6": 78.81758684678512,
        "0": 38.67576734369452,
        		...
        "24": 14.897749534520235
    },
    "RMSE": {
        "-6": 0.028785646228458393,
        "0": 0.013440706733275543,
        		...
        "24": 0.005251956520475109
    },
    "SNRimp": {
        "-6": 20.52452371337197,
        "0": 21.189583201841874,
        		...
        "24": 4.904268340630964
    },
    "SNRo": {
        "-6": 2.751393793207932,
        "0": 9.607560513676567,
        		...
        "24": 17.320105573264037
    },
    "batch_size": 32,
    "macs": 57290752.0,
    "trainable_parmeter": 194689
}
```

---
More arguments of main_CPDAE.py are shown as following:
```console
./ECG_Denoising-CPDAE $ python3 main_CPDAE.py -h
 
 usage: main_CPDAE.py [-h] [-dd DUMP_DIR] [-nd NSTDB_DIR]
                                    [-bs N] [-de DEVICE] [-ep EPOCHS]
                                    [-lr LEARNING_RATE] [-op OPTIMIZER]
                                    [-ds DATA_SHUFFLE] [-lf LOSS_FUNC]
                                    model_name

The execution script for ECG NoiseDB

positional arguments:
  model_name            Module name in dae_model (CPDAE_Lite, CPDAE_Full, or
                        CPDAE_Regular)

optional arguments:
  -h, --help            show this help message and exit
  -dd DUMP_DIR, --dump_dir DUMP_DIR
                        The path to store the model result. (default:
                        ./output)
  -nd NSTDB_DIR, --nstdb_dir NSTDB_DIR
                        The path of nstdb dataset (default:
                        ./dataset/ECG_nstdb_em/raw)
  -bs N, --batch_size N
                        batch size (default: 32)
  -de DEVICE, --device DEVICE
                        The device type of tensor: [cuda, cuda:1, or cpu]
                        (default: cpu)
  -ep EPOCHS, --epochs EPOCHS
                        The number of epochs (default: 500)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Initial learning rate (default: 0.0002)
  -op OPTIMIZER, --optimizer OPTIMIZER
                        The usage optimizer: [Adam, SGD, etc] (default: Adam)
  -ds DATA_SHUFFLE, --data_shuffle DATA_SHUFFLE
                        Random the sequence of the index on training/testing
                        set (default: True)
  -lf LOSS_FUNC, --loss_func LOSS_FUNC
                        The function which would like used in backpropagation
                        (default: MSE)
  -df DUMP_ALL_RESULT_AT_FINAL, --dump_all_result_at_final DUMP_ALL_RESULT_AT_FINAL
                        Dump all the result of test dataset at final, it may
                        take very long time. (default: False)

```
For example:
```console
$ cd <path-of-the-project-ECG_Denoising-CPDAE>
./ECG_Denoising-CPDAE $ python3 main_CPDAE.py CPDAE_Full -de cuda:1 -lr 1e-4 -bs 16 -df True
```
---
## Dump one image with pre-trained model
You can dump a single image via a python script dump_result_1_img.py
```console
./ECG_Denoising-CPDAE $ python3 ./dump_result_1_img.py
```
In the dump_result_1_img.py, there have some parameters can be controlled.

```python
    """parameter"""
    model_name = "CPDAE_Full"
    pwd_model_state_dict = "./output/bs32_CPDAE_Full/model_state_dict.pt"
    pwd_dataset = "./dataset/ECG_nstdb_em/raw"
    normailze = DataNorm.GLOBAL
    fname, ch, index, len = "nstdb_118e00", 0, 118000, 1024
    zero_bias = True
```

## Requirements
### Minimum System Requirements
* OS: Ubuntu 18.04
* Hard Drive: 10 GB Free space
* Memory: 8 GB Free space
* Processor: Dual core CPU
* Python: Python3.6

### Python Virtual Environment
We strongly recommend that install the python package on the python virtual environment, such as: virtualenv, venv.
```console
$ virtualenv -p python3.6 ./virtualenv_cpdae
$ source ./virtualenv/bin/activate
(virtualenv_cpdae) $ <execute-your-python-script>
```

### Python Package Installation

```console
$ pip install -r requirements.txt
```