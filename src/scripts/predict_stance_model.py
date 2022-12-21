import argparse
from common.loadData import load_data
from model.model_stance import StanceModel
import pandas as pd
from common.score import scorePredict
import os


def main(parser):
    args = parser.parse_args()
    test_set = args.test_set
    use_cuda = args.use_cuda
    model_dir_1_stage = args.model_dir_1_stage
    model_dir_2_stage = args.model_dir_2_stage
    model_type = args.model_type
    features_1_stage = args.features_1_stage
    features_2_stage = args.features_2_stage

    label_map = {'unrelated': 3, 'agree': 0, 'disagree': 1, 'discuss': 2}
    df_test_stage_1 = load_data(test_set, features_1_stage, label_map, 'test', '')
    labels_stage_1 = list(df_test_stage_1['labels'].unique())
    labels_stage_1.sort()
    df_test_stage_2 = load_data(test_set, features_2_stage, label_map, 'test', '')
    labels_stage_2 = list(df_test_stage_2['labels'].unique())
    labels_stage_2.sort()
    if model_dir_1_stage != '':
        model = StanceModel(model_type, os.getcwd() + model_dir_1_stage, len(features_1_stage), use_cuda)
        y_predict_1 = model.predict_task(df_test_stage_1)
        df_result = df_test_stage_1
        df_result['predict'] = y_predict_1
        if model_dir_2_stage != '':
            df_y_1 = pd.DataFrame(y_predict_1, columns=['predict'])
            df_y_1_0 = df_y_1[df_y_1['predict'] == 0]
            df_y_1_1 = df_y_1[df_y_1['predict'] == 1]

            p_test_1 = df_test_stage_1.loc[df_y_1_0.index]
            p_test_1['predict'] = df_y_1_0['predict'].values
            p_test_1['predict'] = p_test_1['predict'].replace(0, 3)

            df_test_2 = df_test_stage_1.loc[df_y_1_1.index]
            df_test_2["features"] = df_test_stage_2.loc[df_y_1_1.index]['features']
            model = StanceModel(model_type, os.getcwd() + model_dir_2_stage, len(features_2_stage), use_cuda)
            y_predict_2 = model.predict_task(df_test_2)
            df_test_2['predict'] = y_predict_2
            df_result = pd.concat([p_test_1, df_test_2], axis=0)

    labels = list(df_test_stage_1['labels'].unique())
    labels.sort()
    result, f1 = scorePredict(df_result['predict'].values, df_result['labels'].values, labels)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters

    parser.add_argument("--use_cuda",
                        default=True,
                        action='store_true',
                        help="This parameter should be True if cuda is present.")

    parser.add_argument("--test_set",
                        default="/data/FNC_bert_summary_test_5_features.json",
                        type=str,
                        help="This parameter is the relative dir of test set.")

    parser.add_argument("--model_dir_1_stage",
                        default="/models/",
                        type=str,
                        help="This parameter is the relative dir of the model first stage to predict.")

    parser.add_argument("--model_dir_2_stage",
                        default="/models/",
                        type=str,
                        help="This parameter is the relative dir of the model second stage to predict.")

    parser.add_argument("--model_type",
                        default="roberta",
                        type=str,
                        help="This parameter is the relative type of model to trian and predict.")

    parser.add_argument("--features_1_stage",
                        default=[],
                        nargs='+',
                        help="This parameter is features of model first stage for predict.")

    parser.add_argument("--features_2_stage",
                        default=[],
                        nargs='+',
                        help="This parameter is features of model second stage for predict.")


    main(parser)
