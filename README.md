# An efficient and reproducible deep learning model for musical onset detection

The code aims to reproduce the results in the work:
>Towards an efficient and reproducible deep learning model for musical onset detection

For an interative code demo of this work,
please check our [jupyter notebook](https://goo.gl/Y5KAFC). You should be able to "open with" google colaboratory 
in you google drive, then "open in playground" to execute it block by block.

The code of the demo is in the [colab_demo](https://github.com/musicalOnset-ismir2018/musicalOnset-ismir2018/tree/colab_demo) branch.

## In construction...
- [x] A.1 Code to reproduce experiment results
- [x] A.2 Code to extract log-mel features
- [ ] A.3 Model training code
## Install dependencies
We suggest to install the dependencies in [`virtualenv`](https://virtualenv.pypa.io/en/stable/)
```bash
pip install -r requirements.txt
```
## A. Code usage
### A.1 Reproduce the experiment results with pretrained models
1. Download dataset: [jingju](https://drive.google.com/open?id=17mo5FuWyEHkCFRExKRLGXFcQk2n-jMEW); Böck dataset
is available on request (please send an email).
2. Change `nacta_dataset_root_path`, `nacta2017_dataset_root_path` in `./src/file_path_jingju_shared.py` to
your local jingju dataset path.
3. Change `bock_dataset_root_path` in `./src/file_path_bock.py` to your local Böck dataset path.
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

### A.2 General code for training data extraction
In case that you want to extract the feature, label and sample weights for your own dataset:
1. We assume that your training set audio and annotation are stored in folders `path_audio` and `path_annotation`.
2. Your annotation should conform to either jingju or Böck annotation format. Jingju annotation is stored in
[Praat textgrid file](http://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html). 
In our [jingju textgrid annotations](https://drive.google.com/drive/folders/17mo5FuWyEHkCFRExKRLGXFcQk2n-jMEW?usp=sharing),
two tiers are parsed: `line` and `dianSilence`; The former contains musical line (phrase) level onsets, and the latter
contains syllable level onsets. We assume that you also annotated your audio file in this kind of hierarchical format:
`tier_parent` and `tier_child` corresponding to `line` and `dianSilence`. Böck dataset is annotated at each onset time, 
you can check Böck dataset's annotation in this [link](https://github.com/CPJKU/onset_db),
3. Run below command line to extract training data for your dataset:
```bash
python ./trainingSetFeatureExtraction/training_data_collection_general.py --audio <path_audio> --annotation <path_annotation> --output <path_output> --annotation_type <string, jingju or bock> --phrase <bool> --tier_parent <string e.g. line> --tier_child <string e.g. dianSilence>
```
`--phrase` decides that if you want to extract the feature at file-level. If false is selected, 
you will get a single feature file for the entire input folder.

### A.3 Specific code for jingju and Böck datasets training data extraction
In case the you want to extract the feature, label and sample weights for the jingu and Böck datasets,
we provide the easy executable code for this purpose.
1. Download dataset: [jingju](https://drive.google.com/open?id=17mo5FuWyEHkCFRExKRLGXFcQk2n-jMEW); Böck dataset
is available on request (please send an email).
2. Change `nacta_dataset_root_path`, `nacta2017_dataset_root_path` in `./src/file_path_jingju_shared.py` to
3. Change `bock_dataset_root_path` in `./src/file_path_bock.py` to your local Böck dataset path.
4. Change `feature_data_path` in `./src/file_path_shared.py` to your local output path.
4. Execute below command lines to extract training data for jingju or Böck datasets:
```bash
python ./trainingSetFeatureExtraction/training_data_collection_jingju.py --phrase <bool>
```

```bash
python ./trainingSetFeatureExtraction/training_data_collection_bock.py
```
`--phrase` decides that if you want to extract the feature at file-level. If false is selected, 
you will get a single feature file for the entire input folder. Böck dataset can only be processed
in phrase-level.
your local jingju dataset path.
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
