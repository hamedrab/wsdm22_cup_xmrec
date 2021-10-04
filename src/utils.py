"""
    Some handy functions for pytroch model training ...
"""
import torch
import sys
import math
import pandas as pd
import random
from evaluation import Evaluator


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_evaluations_final(run_mf, test):
    metrics = {'recall_5', 'recall_10', 'recall_20', 'P_5', 'P_10', 'P_20', 'map_cut_10','ndcg_cut_10'}
    eval_obj = Evaluator(metrics)
    indiv_res = eval_obj.evaluate(run_mf, test)
    overall_res = eval_obj.show_all()
    return overall_res, indiv_res


def read_qrel_file(qrel_file):
    qrel = {}
    df_qrel = pd.read_csv(qrel_file, sep="\t")
    for row in df_qrel.itertuples():
        cur_user_qrel = qrel.get(str(row.userId), {})
        cur_user_qrel[str(row.itemId)] = int(row.rating)
        qrel[str(row.userId)] = cur_user_qrel
    return qrel


def write_run_file(rankings, model_output_run):
    with open(model_output_run, 'w') as f:
        f.write(f'userId\titemId\tscore\n')
        for userid, cranks in rankings.items():
            for itemid, score in cranks.items():
                f.write(f'{userid}\t{itemid}\t{score}\n')


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, network.parameters()),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), 
                                                          lr=params['adam_lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, network.parameters()),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer




def get_run_mf(rec_list, unq_users, my_id_bank):
    ranking = {}    
    for cuser in unq_users:
        user_ratings = [x for x in rec_list if x[0]==cuser]
        user_ratings.sort(key=lambda x:x[2], reverse=True)
        ranking[cuser] = user_ratings

    run_mf = {}
    for k, v in ranking.items():
        cur_rank = {}
        for item in v:
            citem_ind = int(item[1])
            citem_id = my_id_bank.query_item_id(citem_ind)
            cur_rank[citem_id]= 2+item[2]
        cuser_ind = int(k)
        cuser_id = my_id_bank.query_user_id(cuser_ind)
        run_mf[cuser_id] = cur_rank
    return run_mf







