import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import pickle
from collections import defaultdict, Counter
from sklearn.metrics import average_precision_score
import tensorflow as tf

def cut_off_dataset(df, frequency=10):

    source_cut_data = df[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
    source_cut = source_cut_data.source_id.value_counts()[(source_cut_data.source_id.value_counts() >= frequency)]
    source_id = np.sort(source_cut.keys())
    df = df.loc[df['source_id'].isin(source_id)]

    return df


def slicing_citation_text(df, number):

    df['#1 String'] = df['left_citated_text'].str[-number:]
    df['#2 String'] = df['right_citated_text'].str[:number]

    return df


def split_dataset(df, year):

    train_idx = df['target_year'][df['target_year'] < year].index
    test_idx = df['target_year'][df['target_year'] >= year].index
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    return train_df, test_df


def get_label(df, train_df, test_df):

    lb = preprocessing.LabelBinarizer()
    lb.fit_transform(df['source_id'].values)
    train_df = convert_argmax(train_df, lb)
    test_df = convert_argmax(test_df, lb)

    return train_df, test_df, lb


def convert_argmax(df, lb):

    y = df['source_id'].values
    y = lb.transform(y)
    y = np.argmax(y, axis=1)
    df['Quality'] = y

    return df


def get_citation_text_id(df):

    left_citated_text = df['#1 String'].values.tolist()
    right_citated_text = df['#2 String'].values.tolist()
    total_citated_text = list(set(left_citated_text + right_citated_text))
    citated_voca = {}
    left_citated_id = []
    right_citated_id = []
    for i, v in enumerate(total_citated_text):
        citated_voca[v] = i
    for l, r in zip(df['#1 String'], df['#2 String']):
        left_citated_id.append(citated_voca[l])
        right_citated_id.append(citated_voca[r])
    df['#1 ID'] = left_citated_id
    df['#2 ID'] = right_citated_id

    return df


def load_data(dataset, column, frequency, seq_len, year, bert_column):

    dataset_name = 'full_context_{}.csv'.format(dataset)
    raw_data_dir = os.path.join(os.getcwd(), 'glue', 'ACRS', dataset_name)
    df = pd.read_csv(raw_data_dir)
    df = df[column]
    df = cut_off_dataset(df, frequency)
    df = slicing_citation_text(df, seq_len)
    df = get_citation_text_id(df)
    train_df, test_df = split_dataset(df, year)
    train_df, test_df, lb = get_label(df, train_df, test_df)
    train_df, test_df = train_df[bert_column], test_df[bert_column]
    return train_df, test_df, lb


def df_to_tsv(train_df, test_df, tsv_dr):

    train_dr = os.path.join(tsv_dr, 'train.tsv')
    test_dr = os.path.join(tsv_dr, 'test.tsv')

    train_df.to_csv(train_dr, sep='\t', index=False)
    test_df.to_csv(test_dr, sep='\t', index=False)


def write_pickle(lb, dr, filename):

    lb_dr = os.path.join(dr, filename)
    with open(lb_dr, 'wb') as handle:
        pickle.dump(lb, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(dr, filename):

    lb_dr = os.path.join(dr, filename)
    pickle_data = pickle.load(open(lb_dr, "rb"))

    return pickle_data


def get_gcn_data(train_df, test_df, save_dir, filename):
    lb_dr = os.path.join(save_dir, filename.split('_')[0], filename)
    with open(lb_dr, 'rb') as f:
        embedding = pickle.load(f)
        node2id = pickle.load(f)
    gcn_train = np.array([[node2id[i]] for i in train_df['target_id'].values])
    gcn_test = np.array([[node2id[i]] for i in test_df['target_id'].values])
    return gcn_train, gcn_test, embedding, node2id


def read_predictions(dr, data_type):
    pred_dr = os.path.join(dr, '{}_results.tsv'.format(data_type))
    f = open(pred_dr, 'r')
    predictions = []
    while True:
        line = f.readline()
        if not line:
            break
        prediction = [float(i) for i in line.split("\t")]
        predictions.append(prediction)
    f.close()

    return np.array(predictions)


def get_multi_label_info(df):

    multi_label_info = []
    for i in df.groupby(['#1 ID', '#2 ID']):
        instance_label = {}
        instance_label['Quality'] = i[1]['Quality'].values
        instance_label['index'] = i[1]['Quality'].index.values
        multi_label_info.append(instance_label)
    return multi_label_info


def convert_class_to_label(multi_label_info, predictions):

    y_true = np.zeros((len(multi_label_info), predictions.shape[1]))
    label_predictions = []
    index = 0
    dummy = []
    for i in multi_label_info:
        label_predictions.append(predictions[i['index'][0]])
        for j in i['Quality']:
            y_true[index][j] = 1
        dummy.append(index)
        index += 1
    return y_true, np.array(label_predictions), dummy


def precision_recall_at_k(cite_id, y_true, predictions, k=10, threshold=0.000001):

    user_est_true = defaultdict(list)
    for uid, true_r, est in zip(cite_id, y_true, predictions):
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        n_rel = len(user_ratings[0][1][user_ratings[0][1] >= threshold])
        n_rec_k = len(user_ratings[0][0][user_ratings[0][0] >= threshold])
        true_value = user_ratings[0][1]
        true_index = true_value.argsort()[::-1]
        true_k = list(filter(lambda i: true_value[i] > threshold, true_index))[:k]
        est_value = user_ratings[0][0]
        est_index = est_value.argsort()[::-1]
        est_k = list(filter(lambda i: est_value[i] > threshold, est_index))[:k]
        count_value = Counter(true_k + est_k)
        n_rel_and_rec_k = list(count_value.values()).count(2)
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    recall_value = sum(list(recalls.values())) / len(recalls)
    precision_value = sum(list(precisions.values())) / len(precisions)

    return precision_value, recall_value


def map_evaluate(cite_id, y_true, y_scores):

    maps = {}
    for id, true, score in zip(cite_id, y_true, y_scores):
        maps[id] = average_precision_score(true, score)

    map_value = sum(list(maps.values()))/len(maps)
    return map_value


# def mrr_evaluate(cite_id, y_true, y_scores):
#
#     # mrr = {}
#     # for id, true, score in zip(cite_id, y_true, y_scores):
#     #     print(true)
#     #     print(score)
#     #     mrr[id] = label_ranking_average_precision_score(true, score)
#     #
#     # mrr_value = sum(list(mrr.values()))/len(mrr)
#     # return mrr_value
#     return label_ranking_average_precision_score(y_true, y_scores)


def write_spec(dr, frequency, text_len, seq_len):

    report_dr = os.path.join(dr, 'f_{}_u_{}_s_{}.txt'.format(frequency, text_len, seq_len))
    with open(report_dr, "w") as f:
        f.write("frequency : {} text_len : {} seq_len : {}\n".format(frequency, text_len, seq_len))


def write_report(dr, performance, top_k=None, method='recall', frequency='error', seq_len='error'):

    if method == 'recall':
        report_contents = 'Recall@{} : {}\n'.format(top_k, performance)
    elif method == 'map':
        report_contents = 'MAP : {}\n'.format(performance)
    else:
        report_contents = 'mrr : {}\n'.format(performance)
    report_dr = os.path.join(dr, 'f_{}_s_{}.txt'.format(frequency, seq_len))
    with open(report_dr, "a") as f:
        f.write(report_contents)


def mean_reciprocal_rank(true_values, predict_values):
    mrr_result = []
    for true, predict in zip(true_values, predict_values):

        idxs = np.where(true == 1)[0]  ## label value 1 else 0

        rank_array = np.argsort(-predict)

        mrr = []
        for idx in idxs:
            rank = np.where(rank_array == idx)[0] + 1

            mrr.append(rank)

        minor_mrr = min(mrr)
        minor_mrr = (1.0 / minor_mrr)
        mrr_result.append(minor_mrr)

    return np.mean(mrr_result)


def get_predictions(result, prediction_dir, dataset_type, num_actual_predict_examples):

    prediction_tsv = "{}_results.tsv".format(dataset_type)
    output_predict_file = os.path.join(prediction_dir, prediction_tsv)
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        num_written_lines = 0
        tf.logging.info("***** Predict results *****")
        for (i, prediction) in enumerate(result):
            probabilities = prediction["probabilities"]
            if i >= num_actual_predict_examples:
                break
            output_line = "\t".join(
                str(class_probability)
                for class_probability in probabilities) + "\n"
            writer.write(output_line)
            num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples

# def change_prediction(experience_dir, dataset_type):


























