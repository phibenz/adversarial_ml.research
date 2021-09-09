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
