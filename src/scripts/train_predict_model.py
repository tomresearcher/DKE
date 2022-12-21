import argparse
from common.loadData import load_data
from model.model_stance import StanceModel
import pandas as pd
from common.score import scorePredict
import wandb
import random
import os


def main(parser):
    args = parser.parse_args()
    training_set = args.training_set
    test_set = args.test_set
    use_cuda = args.use_cuda
    model_dir = args.model_dir
    model_type = args.model_type
    model_name = args.model_name
    type_classify = args.type_classify
    features_stage = args.features_stage
    wandb_project = args.wandb_project
    is_sweeping = args.is_sweeping
    is_evaluate = args.is_evaluate
    best_result_config = args.best_result_config
    exec_model(training_set, test_set, use_cuda, model_dir, model_type, model_name, type_classify, features_stage, wandb_project,
               is_evaluate, is_sweeping, best_result_config)


def exec_model(training_set, test_set, use_cuda, model_dir, model_type, model_name, type_classify, features_stage, wandb_project=None, is_evaluate=False, is_sweeping=False, best_result_config=None):

    if type_classify == 'related':
        label_map = {'unrelated': 0, 'agree': 1, 'disagree': 1, 'discuss': 1}
    elif type_classify == 'stance':
        label_map = {'agree': 0, 'disagree': 1, 'discuss': 2}
    else:
        label_map = {'unrelated': 3, 'agree': 0, 'disagree': 1, 'discuss': 2}

    df_train = load_data(training_set, features_stage, label_map, 'training', type_classify)
    df_test = load_data(test_set, features_stage, label_map, 'test', type_classify)
    labels = list(df_train['labels'].unique())
    labels.sort()
    if model_dir == '':
         model = StanceModel(model_type, model_name, len(features_stage), use_cuda, len(labels), wandb_project, is_sweeping, is_evaluate, best_result_config, True)
         model.train_predict_model(df_train)
    else:
         model = StanceModel(model_type, os.getcwd() + model_dir, len(features_stage), use_cuda)
    y_predict = model.predict_task(df_test)
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_test['labels'].unique())
    labels.sort()
    result, f1 = scorePredict(y_predict, labels_test, labels)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters

    parser.add_argument("--type_classify",
                        default="related",
                        type=str,
                        help="This parameter is used for choose type of classifier (stance, related and all).")

    parser.add_argument("--use_cuda",
                        default=True,
                        action='store_true',
                        help="This parameter should be True if cuda is present.")

    parser.add_argument("--training_set",
                        default="/data/FNC_TR_train.json",
                        type=str,
                        help="This parameter is the relative dir of training set.")

    parser.add_argument("--test_set",
                        default="/data/FNC_TR_test.json",
                        type=str,
                        help="This parameter is the relative dir of test set.")

    parser.add_argument("--model_dir",
                        default='',
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument("--model_type",
                        default="roberta",
                        type=str,
                        help="This parameter is the relative type of model to trian and predict.")

    parser.add_argument("--model_name",
                        default="roberta-large",
                        type=str,
                        help="This parameter is the relative name of model to trian and predict.")

    parser.add_argument("--features_stage",
                        default=[],
                        nargs='+',
                        help="This parameter is the features of model for the each stage for predict.")

    parser.add_argument("--wandb_project",
                        default="",
                        type=str,
                        help="This parameter is the name of wandb project.")

    parser.add_argument("--is_sweeping",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you use sweep search.")

    parser.add_argument("--is_evaluate",
                        default=True,
                        action='store_true',
                        help="This parameter should be True if you want to split train in train and dev.")

    parser.add_argument("--best_result_config",
                        default="",
                        type=str,
                        help="This parameter is the file with best hyperparameters configuration.")


    main(parser)
