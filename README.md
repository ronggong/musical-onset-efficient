# An efficient and reproducible model for musical onset detection

The code aims to reproduce the results in the work:
>Toward an efficient and reproducible deep learning model for musical onset detection
## In construction...
- [x] A.1 Code to reproduce experiment results
- [ ] A.2 Code to extract log-mel features
- [ ] A.3 Model training code
## Install dependencies
We suggest to install the dependencies in [`virtualenv`](https://virtualenv.pypa.io/en/stable/)
```bash
pip install -r requirements.txt
```
## A. Code usage
### A.1 Reproduce the experiment results
1. Download dataset: [jingju](https://drive.google.com/open?id=17mo5FuWyEHkCFRExKRLGXFcQk2n-jMEW); Böck dataset
is available on request (please send an email).
2. change `nacta_dataset_root_path`, `nacta2017_dataset_root_path` in `./src/file_path_jingju_shared.py` to
your local jingju dataset path.
3. change `bock_dataset_root_path` in `./src/file_path_bock.py` to your local Böck dataset path.
4. Download [pretrained models](https://drive.google.com/open?id=1DFB53P4Fz_ixoVFd9fMpW7nvstaK_wuA) and put
them into `./pretrained_models` folder.
4. Execute below command lines to reproduce jingju or Böck datasets results. `archi` variable can be 
chosen from `baseline, relu_dense, no_dense, temporal, bidi_lstms_100, bidi_lstms_200, bidi_lstms_400,
9_layers_cnn, 5_layers_cnn, pretrained, retrained, feature_extractor_a, feature_extractor_b`. Please
read the paper to decide which experiment result you want to reproduce:
```bash
python reproduce_experiment_results.py -d jingju -a archi 
```
```bash
python reproduce_experiment_results.py -d bock -a archi
```
## B. Supplementary information
### B.1 Pretrained models:
[pretrained models link](https://drive.google.com/open?id=1DFB53P4Fz_ixoVFd9fMpW7nvstaK_wuA)

### B.2 Full results (precision, recall, F1):
[full results link](https://drive.google.com/open?id=100RKdVYwsW_WDyd6aDs0YUic84hEdwBl)

### B.3 Statistical significance calculation data
5 times training results for jingju dataset and 8 folds results for Böck dataset.  
[link](https://drive.google.com/open?id=1B1SroQRdsqOjKexA6ICinr3hbPk_jkdZ)

### B.4 Loss curves (section 5.1 in the paper)
These loss curves aim o show the overfitting of Bidi LSTMs 100 and 200 models
 for Böck dataset and 9-layers CNN for both datasets.

Böck dataset Bidi LSTMs 100 losses (fold 2)
![bidi_lstms_100_Bock](figs/loss/bidi_lstms_100_bock.png)

Böck dataset Bidi LSTMs 200 losses (fold 3)
![bidi_lstms_200_Bock](figs/loss/bidi_lstms_200_bock.png)

Böck dataset Bidi LSTMs 400 losses (fold 0)
![bidi_lstms_200_Bock](figs/loss/bidi_lstms_400_bock.png)

Böck dataset baseline and 9-layers CNN losses (model 2)
![9-layers_CNN_and_baseline_bock](figs/loss/9-layers_CNN_bock.png)

Jingju dataset baseline and 9-layers CNN losses (model 2)
![9-layers_CNN_and_baseline_jingju](figs/loss/9-layers_CNN_jingju.png)
