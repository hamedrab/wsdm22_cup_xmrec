import argparse
import os
from os.path import join as pjoin
import pandas as pd
from src.evaluation import Evaluator
import zipfile
import shutil

debug_mode = 0


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def build_ref_pred_pair(ref_dict, pred_dict):
    ref_list, pred_list = [], []
    for k, v in ref_dict.items():
        ref_list.append([v])
        if k in pred_dict:
            pred_list.append(pred_dict[k])
        else:
            pred_list.append(' ')
    return ref_list, pred_list


def get_evaluations_final(run_mf, qrel):
    metrics = {'recall_10', 'ndcg_cut_10'}
    eval_obj = Evaluator(metrics)
    indiv_res = eval_obj.evaluate(run_mf, qrel)
    overall_res = eval_obj.show_all()
    return overall_res, indiv_res


def read_run_file(run_file):
    qret = {}
    df_qret = pd.read_csv(run_file, sep="\t")
    for row in df_qret.itertuples():
        cur_user_qret = qret.get(str(row.userId), {})
        cur_user_qret[str(row.itemId)] = float(row.score)
        qret[str(row.userId)] = cur_user_qret
    return qret


def read_qrel_file(qrel_file):
    qrel = {}
    df_qrel = pd.read_csv(qrel_file, sep="\t")
    for row in df_qrel.itertuples():
        cur_user_qrel = qrel.get(str(row.userId), {})
        cur_user_qrel[str(row.itemId)] = int(row.rating)
        qrel[str(row.userId)] = cur_user_qrel
    return qrel


def read_string(solution_file):
    with open(solution_file) as fi:
        return fi.read().strip()


def merge_run_files(run_dir, market1, market2, output_market):
    predict_path_val_market1 = os.path.join(run_dir, market1, 'valid_pred.tsv')
    predict_path_test_market1 = os.path.join(run_dir, market1, 'test_pred.tsv')

    predict_path_val_market2 = os.path.join(run_dir, market2, 'valid_pred.tsv')
    predict_path_test_market2 = os.path.join(run_dir, market2, 'test_pred.tsv')

    output_market_dir_path = os.path.join(run_dir, output_market)
    mkdir(output_market_dir_path)

    predict_path_val_out = os.path.join(run_dir, output_market, 'valid_pred.tsv')
    predict_path_test_out = os.path.join(run_dir, output_market, 'test_pred.tsv')

    write_market_files(predict_path_val_market1, predict_path_val_market2, predict_path_val_out)
    write_market_files(predict_path_test_market1, predict_path_test_market2, predict_path_test_out)


def write_market_files(predict_path_val_market1, predict_path_val_market2, predict_path_val_out):
    with open(predict_path_val_market1) as fi1:
        with open(predict_path_val_market2) as fi2:
            with open(predict_path_val_out, 'w') as fo:
                for l in fi1:
                    fo.write(l)
                for l in fi2:
                    if not l.startswith('userId'):
                        fo.write(l)


def validate_file_structure(extract_dir):
    for m in ['t1', 't2']:
        for f in ['test_pred.tsv', 'valid_pred.tsv']:
            try:
                with open(os.path.join(extract_dir, m, f)) as fi:
                    pass
            except FileNotFoundError:
                print('{} not found!'.format(os.path.join(extract_dir, m, f)))
                return False
    return True


def get_scores_for_market(input_dir, data_dir, market_name):
    # prepare for val set
    predict_path_val = os.path.join(input_dir, market_name, 'valid_pred.tsv')
    ref_path_val = os.path.join(data_dir, market_name, 'valid_qrel.tsv')
    my_valid_run = read_run_file(predict_path_val)
    my_valid_qrel = read_qrel_file(ref_path_val)
    task_ov_val, task_ind_val = get_evaluations_final(my_valid_run, my_valid_qrel)

    return task_ov_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_file", help="Zip file that contains the run files to be submitted to Codalab.")
    parser.add_argument("--data_dir", help="Path to the DATA dir of the kit. Default: ./DATA/.", default='./DATA/')
    args = parser.parse_args()

    extract_dir = './tmp/'

    scores = ['ndcg_cut_10', 'recall_10']

    score_names = {
        'recall_10': {'val': 'r10_val', 'test': 'r10_test'},
        'ndcg_cut_10': {'val': 'ndcg10_val', 'test': 'ndcg10_test'}
    }

    # We assume that the submission comes with two markets (i.e., t1 and t2).
    marekts = ['t1', 't2', 't1t2']

    # First we unzip the run file in a tmp folder then start evaluating it.
    mkdir(extract_dir)

    print('Extracting the submission zip file')

    with zipfile.ZipFile(args.submission_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print('Validating the file structure of the submission')

    file_structure_validation = validate_file_structure(extract_dir)

    if file_structure_validation:
        print('File structure validation successfully passed')
    else:
        print('File structure validation failed. Please refer to the instructions')
        return

    print('Evaluating the validation set')

    # Then we merge the run files of the two markets for the joint performance evaluation, and call it 't1t2'.
    merge_run_files(extract_dir, 't1', 't2', 't1t2')

    task_ov_test, task_ov_val = {}, {}

    for m in marekts:  # iterate over the three target markets (including the joint one)
        print(
            "===================== Market : " + m + "=====================")
        task_ov_val[m] = get_scores_for_market(extract_dir, args.data_dir, m)
        for score in scores:  # iterating over the scores
            score_val_name = score_names[score]['val']
            score_val = task_ov_val[m][score]
            print(
                "======= Set val : score(" + score_val_name + ")=%0.12f =======" % score_val)

    # remove the tmp directory
    shutil.rmtree(extract_dir)


if __name__ == "__main__":
    main()
