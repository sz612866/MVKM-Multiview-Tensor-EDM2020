# import sys
#
# sys.path.append("../")
from data_helper import *
from multiview import MultiView
from multiprocessing import Pool, Lock
import pickle
import json
import os
import pathlib
import numpy as np

output_lock = Lock()


def single_multiview_exp(data_str, course_num, model_str, fold, skill_dim, concept_dim, lambda_s,
                         lambda_t, lambda_q, lambda_e, lambda_bias, penalty_weight, markovian,
                         trade_off_example, max_iter, lr, log_file):
    """
    pipeline of running single experiment with 5-fold cross validation
    :para: a list of parameters for a single case of experiment
    :return:
    """

    with open('data/{}/{}/{}_train_test.pkl'.format(data_str, course_num, fold), 'rb') as f:
        data = pickle.load(f)
        print(data.keys())

    model_config = config(data, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q, lambda_bias,
                          penalty_weight, markovian_steps=markovian, lambda_e=lambda_e, lr=lr,
                          max_iter=max_iter, trade_off_example=trade_off_example, log_file=log_file)
    test_set = model_config['test']
    if model_str == 'multiview':
        model = MultiView(model_config)
    else:
        raise EnvironmentError("ERROR!!")

    print(model.train_data)
    # find the first testing attempt, and add all examples before test_start_attempt into train_data
    test_start_attempt = None
    for (stud, att, index, obs, res) in sorted(test_set, key=lambda x: x[1]):
        if res == 0:
            test_start_attempt = att
            break
        else:
            model.train_data.append((stud, att, index, obs, res))
    if None == test_start_attempt:
        raise EnvironmentError

    total_test_count = 0
    sum_square_error, sum_abs_error = [0.] * 2
    for attempt in range(test_start_attempt, model.num_attempts):
        # train, and then predict the obs at current attempt
        model.current_test_attempt = attempt
        model.lr = lr
        model.training()
        test_data = []
        for (stud, att, index, obs, res) in test_set:
            if att == model.current_test_attempt:
                test_data.append((stud, att, index, obs, res))
                model.train_data.append((stud, att, index, obs, res))

                for i in range(max(0, att - model.markovian_steps), att):
                    model.train_data_markovian.append((stud, i, index, res))
                upper_steps = min(model.num_attempts, attempt + model.markovian_steps + 1)
                for j in range(attempt + 1, upper_steps):
                    model.train_data_markovian.append((stud, j, index, res))

        test_perf = model.testing(test_data)
        # re-initialize the bias for each attempt, student, question, example
        if attempt != model.num_attempts - 1:
            model.bias_s = np.zeros(model.num_users)
            model.bias_q = np.zeros(model.num_questions)
            model.bias_a = np.zeros(model.num_attempts)

        # if attempt not in perf_dict[fold]:
        #     perf_dict[fold][attempt] = test_perf
        test_count, _rmse, _mae = test_perf

        # cumulative all metrics over all attempts
        sum_square_error += (_rmse ** 2) * test_count
        sum_abs_error += _mae * test_count
        total_test_count += test_count

    rmse = np.sqrt(sum_square_error / total_test_count)
    mae = sum_abs_error / total_test_count

    dir_path = "saved_models/{}/{}/{}/".format(data_str, course_num, model_str)
    if not os.path.exists(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    file_name = "fold_{}_skill_{}_concept_{}_ls_{}_lt_{}_lq_{}_le_{}_lbias_{}_pw_{}_" \
                "markov_{}_tradeoff_{}_lr_{}_max_iter_{}_model.pkl".format(fold, skill_dim,
                                                                           concept_dim, lambda_s,
                                                                           lambda_t, lambda_q,
                                                                           lambda_e, lambda_bias,
                                                                           penalty_weight,
                                                                           markovian,
                                                                           trade_off_example, lr,
                                                                           max_iter)

    file_path = dir_path + file_name
    pickle.dump(model, open(file_path, "wb"))
    return [fold, total_test_count, rmse, mae]


def collect_results(data_str, course_num, model_str):
    # def model_analysis(data_str, course_num, model_str, fold):
    output_path = "results/{}/{}/{}/final_exp_results.csv".format(data_str, course_num, model_str)
    with open(output_path, "r") as f:
        for line in f:
            result = json.loads(line)
            print('skill_dim: {skill_dim}, concept_dim: {concept_dim}, lambda_s: {lambda_s}, '
                  'lambda_t: {lambda_t}, lambda_q: {lambda_q}, lambda_e: {lambda_e}, '
                  'lambda_bias:{lambda_bias}, penalty_weight:{penalty_weight}, markovian: '
                  '{markovian_steps}, trade_off: {trade_off}'.format(**result))

            skill_dim = result['skill_dim']
            concept_dim = result['concept_dim']
            lambda_s = result['lambda_s']
            lambda_t = result['lambda_t']
            lambda_q = result['lambda_q']
            lambda_e = result['lambda_e']
            lambda_bias = result['lambda_bias']
            penalty_weight = result['penalty_weight']
            markovian = result['markovian_steps']
            trade_off = result['trade_off']
            lr = result['lr']
            max_iter = result['max_iter']

            rmse_list = []
            for _fold in result['perf'].keys():
                # for v in result['perf'][_fold]['overall']:
                rmse = result['perf'][_fold][1]
                rmse_list.append(rmse)
                for v in result['perf'][_fold]:
                    print("{},".format(v), end="")
                print("")
            print('average rmse {}'.format(np.mean(rmse_list)))


def run_morf():
    data_str = 'morf'
    course_num = 'Quiz_Discussion'
    model_str = 'multiview'

    # run the single test with best hyperparameters for Quiz-Discussion dataset
    skill_dim = 39
    concept_dim = 5
    lambda_s = 0
    lambda_t = 0
    lambda_q = 0
    lambda_e = 0
    lambda_bias = 0
    penalty_weight = 1.0
    markovian = 1
    trade_off = 0.05
    lr = 0.1
    max_iter = 50

    dir_path = "results/{}/{}/{}".format(data_str, course_num, model_str)
    if not os.path.exists(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    output_path = "{}/final_exp_results.csv".format(dir_path)
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            pass

    log_path = "logs/{}/{}/{}".format(data_str, course_num, model_str)
    if not os.path.exists(log_path):
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)



    # process the 5-fold cross validation in parallel
    para_list = []
    results = []
    for fold in range(1, 6):
        log_file = '{}/fold_{}_skill_{}_concept_{}_ls_{}_lt_{}_lq_{}_le_{}_lbias_{}_pw_{}_' \
                   'markov_{}_trade_off_{}_lr_{}_iter_{}'.format(log_path, fold, skill_dim,
                    concept_dim, lambda_s, lambda_t, lambda_q, lambda_e, lambda_bias,
                    penalty_weight, markovian, trade_off, lr, max_iter)
        para = (data_str, course_num, model_str, fold, skill_dim, concept_dim, lambda_s, lambda_t,
                lambda_q, lambda_e, lambda_bias, penalty_weight, markovian, trade_off, max_iter, lr,
                log_file)
        para_list.append(para)
        # result = single_multiview_exp(*para)
        # results.append(result)
    pool = Pool(processes=5)
    results = pool.starmap(single_multiview_exp, para_list)
    pool.close()

    # collect and store the results of 5-folds
    perf_dict = {}
    for result in results:
        fold = result[0]
        print(result)
        perf_dict[fold] = result[1:]
    result = {'skill_dim': skill_dim,
              'concept_dim': concept_dim,
              'lambda_s': lambda_s,
              'lambda_t': lambda_t,
              'lambda_q': lambda_q,
              'lambda_e': lambda_e,
              'lambda_bias': lambda_bias,
              'penalty_weight': penalty_weight,
              'markovian_steps': markovian,
              'trade_off': trade_off,
              'lr': lr,
              'max_iter': max_iter,
              'perf': perf_dict}
    # save the performance result in the file
    output_lock.acquire()
    with open(output_path, "a") as output_file:
        output_file.write(json.dumps(result) + '\n')
    output_lock.release()

    # read and show the performance result
    collect_results(data_str, course_num, model_str)


if __name__ == '__main__':
    run_morf()
