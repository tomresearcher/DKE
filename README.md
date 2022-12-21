#Paper DKE

**(2022/11/10) create readme **

This repository contains the code of the main experiments of our research using the Fake News Challenge dataset (FNC). The task consists of classifying a given headline with respect to its body-text (agree, disagree, discuss, and unrelated). 
Our approach is based on a neural network that uses automatic summaries generated from the body-text, so it is recommended to execute the code in a GPU device.

The code to train the models together with the generated summaries are available. In addition, our pre-trained models are also available. Please follow the instructions below, considering that some differences in the results could be due to library versions and the features of the devices where the code is executed.

You can use this code in your host environment or in a docker container.
### Requirements
* Python >= 3.7
* Pytorch >= 1.12
* Transformers >= 4.12.5
* Linux OS or Docker

### Download dataset
Download the FNC-dataset and the generatic summaries from this link:
```bash
wget -O data.zip "https://drive.google.com/uc?export=download&id=1H1do0h6R__QdZKj42fG36tMUX8DQW1io"
unzip data.zip
rm data.zip
```

### Download fine-tuned models

If you want to predict with our models you will download this folder.
Download pre-training models from this google account:

```bash
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id=1ZQnJOVhiK71BD6-LCpDDGHJCCob7XH4u' -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O models_DKE.zip \
     'https://docs.google.com/uc?export=download&id=1ZQnJOVhiK71BD6-LCpDDGHJCCob7XH4u&confirm='$(<confirm.txt)
unzip models_DKE.zip
rm models_DKE.zip
rm confirm.txt
rm cookies.txt
```

### Installation in docker container
You only need to have docker installed. 
 
Docker images tested:
* pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

You need to grant permission to install_docker.sh file:
```bash
chmod 777 install_docker.sh
```

If you have GPU you will use this command:
```bash
docker run --name name_container -it --net=host --gpus device=device_number -v folder_dir_with_code:/workspace pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel bash -c "./install_docker.sh"
```

If you have not GPU you will use this command:
```bash
docker run --name name_container -it --net=host -v folder_dir_with_code:/workspace pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel bash -c "./install_docker.sh"
```

These commands should be executed into the code folder.
### Description of scripts

##### Train and predict models
If you want train and predict your model you will use train_predict_model.py

These parameters allow to configure the system to train or predict.

|Field| Description                                                                         |
|---|-------------------------------------------------------------------------------------|
|type_classify| This parameter is used to choose the type of classifier (stance, related and all).  |
|use_cuda| This parameter can be used if cuda is present.                                      |
|training_set| This parameter is the relative directory of the training set.                       |
|test_set| This parameter is the relative directory of the test set.                           |
|model_dir| This parameter is the relative directory of the model for prediction.               |
|model_type| This parameter is the relative type of model to trian and predict.                  |    
|model_name| This parameter is the relative name of model to trian and predict.                  |
|features_stage| This parameter contains the features of the model for the each stage of prediction: related ('cosine_similarity', 'max_score_in_position', 'overlap', 'soft_cosine_similarity', 'bert_cosine_similarity', hellinger_score, kullback_leibler_score) and stance (polarity_head_neg', 'polarity_head_pos', 'polarity_sum_neg', 'polarity_sum_pos', 'polarity_head_neu', 'polarity_sum_neu'). |
|wandb_project| This parameter is the name of wandb project.                                                                                                                                                                                                                                                                                                                                                       |
|is_sweeping| This parameter should be True if you use sweep search.                                                                                                                                                                                                                                                                                                                                               |
|is_evaluate| This parameter should be True if you want to split train in train and dev.                                                                                                                                                                                                                                                                                                                                              |
|best_result_config| This parameter is the file with best hyperparameters configuration.                                                                                                                                                                                                                                                                                                                                               |

You can create an account at https://wandb.ai/ to monitor your training. 

For example, if you want to train and predict "stance" as the type of classifier:
```bash
--type_classify 'stance' --features_stage 'polarity_head_neg' 'polarity_head_pos' 'polarity_sum_neg' 'polarity_sum_pos' 'polarity_head_neu' 'polarity_sum_neu'
```
For example, if you want to train and predict "related" as the type of classifier with different features:
```bash
--type_classify 'related' --features_stage 'cosine_similarity' 'max_score_in_position' 'overlap' 'soft_cosine_similarity' 'bert_cosine_similarity'
```

Execute this command to train and predict "related" as the type of classifier with different features.

```bash
PYTHONPATH=src python src/scripts/train_predict_model.py --training_set "/data/FNC_TR_train.json" --test_set "/data/FNC_TR_test.json" --type_classify 'related' --features_stage 'cosine_similarity' 'max_score_in_position' 'overlap' 'soft_cosine_similarity' 'bert_cosine_similarity' --use_cuda
```

Execute this command to predict "related" as the type of classifier with different features using our pre-trained model
```bash
PYTHONPATH=src python src/scripts/train_predict_model.py --model_dir "/model" --test_set "/data/FNC_TR_test.json" --type_classify 'related' --features_stage 'cosine_similarity', 'max_score_in_position', 'overlap', 'soft_cosine_similarity', 'bert_cosine_similarity' --use_cuda
```

##### Predicting using the whole architecture  
If you want to predict all models, you will use predict_stance_model.py

These parameters allow to configure the system to obtain the prediction with one stage or with two stages.

| Field             | Description                                                                                                                                                                                                      |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| use_cuda          | This parameter can be used if cuda is present.                                                                                                                                                                   |
| test_set          | This parameter is the relative directory of the test set.                                                                                                                                                        |
| model_dir_1_stage | This parameter is the relative directory of the model for predicting the first stage, i.e., related and unrelated.                                                                                               |
| model_dir_2_stage | This parameter is the relative directory of the model for predicting the second stage, i.e., agree, disagree, and discuss.                                                                                       |
| features_1_stage  | This parameter contains the features of the model for the first stage of prediction ('cosine_similarity', 'max_score_in_position', 'overlap', 'soft_cosine_similarity', 'bert_cosine_similarity').               |
| features_2_stage  | This parameter contains the features of the model for the second stage of prediction ('polarity_head_neg', 'polarity_head_pos', 'polarity_sum_neg', 'polarity_sum_pos', 'polarity_head_neu', 'polarity_sum_neu') |

Execute this command to predict the FNC classes with your models 
```bash
PYTHONPATH=src python src/scripts/predict_stance_model.py --model_dir_1_stage "/model_1" --model_dir_2_stage "/model_2" --test_set "/data/FNC_TR_test.json" --features_1_stage  --use_cuda
```
Execute this command to predict the FNC classes with our provided pre-trained models
```bash
PYTHONPATH=src python src/scripts/predict_stance_model.py --model_dir_1_stage "/models/related" --model_dir_2_stage "/models/stance" --test_set "/data/FNC_TR_test.json" --features_1_stage 'cosine_similarity' 'max_score_in_position' 'overlap' 'soft_cosine_similarity' 'bert_cosine_similarity' 
--features_2_stage 'polarity_head_neg' 'polarity_head_pos' 'polarity_sum_neg' 'polarity_sum_pos' 'polarity_head_neu' 'polarity_sum_neu' --use_cuda
```
Note: If you don't have GPU remove "--use_cuda" in the commands


Related features: 'cosine_similarity' 'max_score_in_position' 'overlap' 'soft_cosine_similarity' 'bert_cosine_similarity'

Stance features: 'polarity_head_neg' 'polarity_head_pos' 'polarity_sum_neg' 'polarity_sum_pos' 'polarity_head_neu' 'polarity_sum_neu'

##### Optimize hyperparameters

In addition, you can use this library to optimize hyperparameters. We found that the seed of initialization affects the training process, then we include it in the hyperparameter optimization. For example:

``` bash
method: bayes
metric:
  goal: maximize
  name: f1
parameters:
  dropout:
    values: [0.1, 0.2, 0.3]
  batch_size:
    values: [2, 4, 8]
  learning_rate:
    distribution: uniform
    max: 5e-05
    min: 1e-05
  num_train_epochs:
    max: 4
    min: 2
  weight_decay:
    distribution: uniform
    max: 0.3
    min: 0
program: sweep_roberta.py

```
You can consult https://docs.wandb.ai/guides/sweeps/quickstart to create sweeps. In sweep_roberta.py you have an example of sweep execution. 

##### Result of hyperparameters optimization

In sweep_results folder you can find the configuration of each trained model

TextRank:
Relatedness stage: apricot-water-149.csv
Stance stage: wise-smoke-151.csv

BERT:
Relatedness stage: comfy-snowball-2.csv
Stance stage: copper-totem-188.csv

BART:
Relatedness stage: wise-smoke-151.csv
Stance stage: wise-smoke-151.csv


### Contacts:
email: rsepulveda911112@gmail.com

### Citation:

  
### License:
  * Apache License Version 2.0 