# Reproduce the resutls of "Adversarial Examples are not bugs, they are features" [[ArXiv]](https://export.arxiv.org/pdf/1905.02175) [[Github]](https://github.com/MadryLab/robustness)

## Evaluate the pretrained model checkpoints
First, lets evaluate the pretrained models provided by the [original repo](https://github.com/MadryLab/robustness)

### Download the pretrained model checkpoints
Ensure that you are in che `./code` folder and run `bash reproduce_features_not_bugs/download_models_fnb.sh`

### Convert the models
Ensure that you are in che `./code` folder and run `bash reproduce_features_not_bugs/convert_models_fnb.sh`

### Evaluate the models
Ensure that you are in che `./code` folder and run `bash reproduce_features_not_bugs/evaluate_models_fnb.sh`. This should lead to the following resutls. 
Note the following: 
* adapt the `--model-path` variables in `evaluate_models_fnb.sh` accordingly.
* All evaluations were made with a number of 20 steps. 
* Theresults are comparabale with those from the original repository.

#### CIFAR10 ResNet50 Adversarially Trained and Evaluated with PGD L2

| ε-test | 0     | 0.25  | 0.5   | 1.0   |
|--------|-------|-------|-------|-------|
| 0      | 95.25 | 92.77 | 90.83 | 81.62 |
| 0.25   | 8.68  | 81.21 | 82.40 | 75.53 |
| 0.5    | 0.29  | 62.29 | 70.17 | 68.63 |
| 1.0    | 0.00  | 21.18 | 40.48 | 52.72 |
| 2.0    | 0.00  | 0.59  | 5.24  | 18.59 |

#### CIFAR10 ResNet50 Adversarially Trained and Evaluated with PGD L2 

| ε-test | 0     | 8/255 |
|--------|-------|-------|
| 0      | 95.25 | 81.62 |
| 8/255  | 0.00  | 80.94 |
| 16/255 | 0.00  | 80.25 |

#### ImageNet ResNet50 Adversarially Trained and Evaluated with PGD L2

| ε-test | 0     | 3.0   |
|--------|-------|-------|
| 0      | 76.13 | 57.90 |
| 0.5    | 3.35  | 54.42 |
| 1.0    | 0.44  | 50.67 |
| 2.0    | 0.16  | 43.04 |
| 3.0    | 0.13  | 35.16 |

#### ImageNet ResNet50 Adversarially Trained and Evaluated with PGD Linf

| ε-test | 0     | 4/255 | 8/255 |
|--------|-------|-------|-------|
| 0      | 76.13 | 62.42 | 47.91 |
| 4/255  | 0.04  | 33.57 | 33.06 |
| 8/255  | 0.02  | 13.15 | 19.64 |
| 16/255 | 0.01  | 1.53  | 4.99  |


## Evaluate the provided datasets
### Download the provided datasets 
Download the provided datasets with `bash ./reproduce_features_not_bugs/download_datasets.sh`.

### Train ResNet50 on the provided datasets
Run `bash ./reproduce_features_not_bugs/train_datasets_fnb.sh`.

### Evaluate the trained ResNet50 models
Run `bash ./reproduce_features_not_bugs/eval_datasets_fnb.sh`. The following results were obtained with the above scripts, which are comparable to Table 7 and Table 1 of the original paper.

| Dataset            | 0     | 0.25  | 0.5   |
|--------------------|-------|-------|-------|
| Non-robust dataset | 86.28 | 0.03  | 0.0   |
| Robust dataset     | 84.55 | 46.54 | 16.62 |
| D rand             | 64.13 | 0.00  | 0.00  |
| D det              | 43.54 | 0.00  | 0.00  |

## Reproduce the extraction of datasets 
### Extract the datasets
Run `bash ./reproduce_features_not_bugs/extract_grad_datasets.sh`. The respective gradient images are stored in the folder of the model, they were extracted from. 

### Train ResNet50 on the extracted datasets
Run `bash ./reproduce_features_not_bugs/train_grad_datasets.sh`.

### Evaluate the trained ResNet50 models
Run `bash ./reproduce_features_not_bugs/evaluate_models_grad_imgs.sh`. The following obtained results are comparable to the above Table. 

| Dataset            | 0     | 0.25  | 0.5   |
|--------------------|-------|-------|-------|
| Non-robust dataset | 84.38 | 0.02  | 0.0   |
| Robust dataset     | 86.79 | 49.93 | 19.22 |
| D rand             | 67.75 | 0.00  | 0.00  |
| D det              | 42.03 | 0.00  | 0.00  |
