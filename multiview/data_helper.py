def config(data, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q, lambda_bias,
           penalty_weight, markovian_steps=1, lambda_e=0., tol=1e-3, trade_off_example=0.,
           max_iter=30, lr=0.01, log_file=None, *args):
    """
    generate model configurations for training and testing
    such as initialization of each parameters and hyperparameters
    :param data: a preprocessed formated data dictionary
    :param data_resource: if None, we only use data from all resourcees;
    :param tol: torelence rate when doing SGD
    :param trade_off: trade-off of training between two different resources
    :return: config dict
    """
    config = {
        'num_users': data['num_users'],
        'num_questions': data['num_quizs'],
        'num_discussions': data['num_disicussions'],
        'num_attempts': data['num_attempts'],
        'num_skills': skill_dim,
        'num_concepts': concept_dim,
        'lambda_s': lambda_s,
        'lambda_t': lambda_t,
        'lambda_q': lambda_q,
        'lambda_e': lambda_e,
        'lambda_bias': lambda_bias,
        'penalty_weight': penalty_weight,
        'markovian_steps': markovian_steps,
        'trade_off_example': trade_off_example,
        'lr': lr,
        'tol': tol,
        'max_iter': max_iter,
        'log_file': log_file}
    print(config)
    # generate config, train_set, test_set for general train and test
    train_set = []
    for (stud, ques, obs, att, res) in data['train']:
        if int(att) < 100:
            train_set.append((int(stud), int(att), int(ques), float(obs), int(res)))
    config['train'] = train_set
    test_set = []
    test_users = {}
    for (stud, ques, obs, att, res) in data['test']:
        if int(att) < 100:
            test_set.append((int(stud), int(att), int(ques), float(obs), int(res)))
            if stud not in test_users:
                test_users[stud] = 1
    config['num_attempts'] = 100
    config['test'] = test_set
    config['test_users'] = test_users
    return config
