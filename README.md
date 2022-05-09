## dl4h: Paper Reproduction Project

Original Paper:
>Ma, F., Wang, Y., Xiao, H. et al. Incorporating medical code descriptions for diagnosis prediction in healthcare. BMC Med Inform Decis Mak 19, 267 (2019). https://doi.org/10.1186/s12911-019-0961-2

### Setup
The code in this repository uses Python 3.8 and PyTorch 1.10.

It is highly recommended to create a Python virtual environment with Python 3.8 and then execute the following:

`pip install -r requirements.txt`


### Data Download
This project uses the publicly available [MIMIC-III dataset](https://mimic.mit.edu/docs/iii/) as well as the fastText pre-trained English word vectors linked [here](https://fasttext.cc/docs/en/english-vectors.html).

From the fastText link above, download the file `crawl-300d-2M-subword.zip` and store it in the `data` dir of the locally cloned repo.

You will need to provide a local path to the downloaded MIMIC-III dataset which contains the individual .csv files.

### Data Pre-processing
The pre-processed data is provided for you in Python `pickle` format in the dir `data`. These files are simply lists of output to be loaded for training the various models. This is done to save time in testing the code since the data pre-processing takes about 65 mins to complete.

To run data pre-processing from scratch the steps are as follows:
1. Launch jupyter notebook from the root directory of the cloned repo: `jupyter notebook .`
2. Open the jupyter notebook `DataProcessing.ipynb`
3. In the 2nd cell, set the variable `MIMIC_DATA_PATH` to your local path to the MIMIC-III dataset
4. Execute all cells in the notebook
5. Output data will be saved to `data` folder in your local git repository

### Training and evaluating the models
There are five baseline models and five enhanced models, as well as the CNN model for learning the embedding matrix of  diagnosis code descriptions. All models are provided as standalone jupyter notebooks that can be trained independently provided the necessary data is available. It is recommended to first train the CNN, and then train the baseline and ehanced models.

To train the CNN make sure you have first downloaded the pre-trained fastText word vector model and extracted the .zip archive in the `data` folder if your locally cloned repository. The name of the file should be `crawl-300d-2M-subword.bin`.

To train the model:
1. Launch jupyter notebook from the root directory of the cloned repo: `jupyter notebook .`
2. Open the jupyter notebook `EmbeddingCNN.ipynb`
3. Execute all cells to train and evaluate the model
4. The trained model will be saved to the `models` dir
5. The trained embedding vector **E** in the `data` dir as `embedding_matrix.pt` to be used in all enhanced models

***Note***: The pre-trained EmbeddingCNN model is provided in the `models` dir, as well as the pre-trained embedding matrix in the `data` dir.

To train the baseline and ehnaced models, the processs is similar:
1. Launch jupyter notebook from the root directory of the cloned repo: `jupyter notebook .`
2. Open the jupyter notebook for the desired model (i.e. `BaselineDipole.ipynb`)
3. Execute all cells in the notebook to train and evaluate the model
4. The trained model will be saved to the `models` dir

***Note***: All models have a pre-trained model provided which can be loaded and evaluated to bypass training.

To load the pre-trained models:
1. Execute all cells in the desired notebook ***before*** the cell titled `Set num epochs and train model`
2. Skip to the final cell titled `Load pre-trained model and evaluate` and execute this cell


### Overview of results

***precision@k***
| @k | MLP    | MLP+   | RNN    | RNN+   | RNNa   | RNNa+  | Dipole | Dipole+ | RETAIN | RETAIN+ |
|----|--------|--------|--------|--------|--------|--------|--------|---------|--------|---------|
| 5  | 0.6879 | 0.6934 | 0.7156 | 0.7111 | 0.7390 | 0.7115 | 0.7237 | 0.7217  | 0.7135 | 0.6896  |
| 10 | 0.6526 | 0.6505 | 0.6787 | 0.6585 | 0.6931 | 0.6742 | 0.6858 | 0.6820  | 0.6651 | 0.6472  |
| 15 | 0.6849 | 0.6789 | 0.7060 | 0.6932 | 0.7237 | 0.7052 | 0.7164 | 0.7164  | 0.6952 | 0.6867  |
| 20 | 0.7417 | 0.7332 | 0.7606 | 0.7484 | 0.7733 | 0.7601 | 0.7716 | 0.7709  | 0.7476 | 0.7440  |
| 25 | 0.7922 | 0.7856 | 0.8063 | 0.7959 | 0.8183 | 0.8096 | 0.8206 | 0.8154  | 0.7977 | 0.7936  |
| 30 | 0.8375 | 0.8299 | 0.8454 | 0.8348 | 0.8554 | 0.8451 | 0.8533 | 0.8503  | 0.8366 | 0.8325  |

***accuracy@k***

| @k | MLP    | MLP+   | RNN    | RNN+   | RNNa   | RNNa+  | Dipole | Dipole+ | RETAIN | RETAIN+ |
|----|--------|--------|--------|--------|--------|--------|--------|---------|--------|---------|
| 5  | 0.3490 | 0.3608 | 0.3851 | 0.3860 | 0.3969 | 0.3856 | 0.3954 | 0.3937  | 0.3766 | 0.3794  |
| 10 | 0.5466 | 0.5467 | 0.5821 | 0.5675 | 0.5886 | 0.5775 | 0.5859 | 0.5812  | 0.5660 | 0.5581  |
| 15 | 0.6604 | 0.6548 | 0.6850 | 0.6732 | 0.7001 | 0.6845 | 0.6938 | 0.6899  | 0.6730 | 0.6666  |
| 20 | 0.7379 | 0.7923 | 0.7572 | 0.7455 | 0.7696 | 0.7572 | 0.7685 | 0.7651  | 0.7439 | 0.7409  |
| 25 | 0.7920 | 0.7852 | 0.8061 | 0.7958 | 0.8182 | 0.8095 | 0.8204 | 0.8148  | 0.7973 | 0.7934  |
| 30 | 0.8375 | 0.8299 | 0.8454 | 0.8348 | 0.8554 | 0.8451 | 0.8533 | 0.8503  | 0.8366 | 0.8325  |


UIUC CS598 Final Project
