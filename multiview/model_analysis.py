import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy import stats


def model_analysis(data_str, course_num, model_str, fold, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q,
                   lambda_e, lambda_bias, lr, penalty_weight, markovian, max_iter, trade_off):
    model_path = "outputs/{}/{}_{}_fold_{}_skill_{}_concept_{}_ls_{}_lt_{}_lq_{}_le_{}_lbias_{}_lr_{}_pw_{}_markov" \
                 "_{}_iter_{}_trade_off_{}_model.pkl".format(
        data_str, course_num, model_str, fold, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q, lambda_e,
        lambda_bias, lr, penalty_weight, markovian, max_iter, trade_off)


    # true_T = pickle.load(open("data/{}/{}/simulated_T.pkl".format(data_str, course_num), "rb"))['t']
    # model_path = "outputs/{}/{}_{}_fold_{}_model.pkl".format(data_str, course_num, model_str, fold)
    model = pickle.load(open(model_path, "rb"))
    initial_model_path = "outputs/{}/{}_{}_initial_model.pkl".format(data_str, course_num, model_str)
    initial_model = pickle.load(open(initial_model_path, "rb"))
    # train_set = np.array(initial_model.train_data)
    # print("shape of T {}".format(model.T.shape))
    print(model_path)

    # used for generating figures
    # color_dict = {7: 'darkorange', 0: 'b', 6: 'g', 4: 'r', 3: 'm', 5: 'c', 2: 'y', 1: 'k', 8: 'slategrey'}
    # marker_dict = {0: 'D', 1: 'X', 2: '^', 4: 's', 3: 'P', 6: '<', 5: '*', 7: '>', 8: 'h'}
    # line_style_dict = {0: '-.', 1: '--', 2: ':', 3: '-'}
    color_dict = {0: 'darkorange', 1: 'b', 2: 'g', 3: 'r', 4: 'm', 5: 'c', 6: 'y', 7: 'k', 8: 'slategrey'}
    marker_dict = {0: 'D', 1: 'X', 2: '^', 3: 's', 4: 'P', 5: '<', 6: '*', 7: '>', 8: 'h'}
    line_style_dict = {0: '-.', 1: '--', 2: ':', 3: '-'}

    print("===step 0: output model statistics===")
    # compute difference between the current model and the initial model
    attempt_question_dict = {}
    for (student, attempt, question, obs, resource) in model.train_data:
        # if resource == 0:
        if attempt not in attempt_question_dict:
            attempt_question_dict[attempt] = {}
        if question not in attempt_question_dict[attempt]:
            attempt_question_dict[attempt][question] = [1, obs]
        else:
            # update the count and mean score
            count = attempt_question_dict[attempt][question][0]
            mean_obs = attempt_question_dict[attempt][question][1]
            sum = mean_obs * count
            count += 1
            mean_obs = (sum + obs) / count
            attempt_question_dict[attempt][question][0] = count
            attempt_question_dict[attempt][question][1] = mean_obs

    ST = np.zeros((model.num_users, model.num_attempts, model.num_concepts))
    STQ = np.zeros((model.num_users, model.num_attempts, model.num_questions))
    T_plus_bias_a = np.zeros_like(model.T)
    ST_mean = []
    for attempt in range(model.num_attempts):
        T_plus_bias_a[:, attempt, :] = model.T[:, attempt, :] + model.bias_a[attempt]
        ST[:, attempt, :] = np.dot(model.S, model.T[:, attempt, :])
        ST_mean.append(np.mean(ST[:, attempt, :]))
        STQ[:, attempt, :] = np.dot(np.dot(model.S, model.T[:, attempt, :]), model.Q)

    for attempt in range(model.num_attempts):
        question_count_tuples = sorted(attempt_question_dict[attempt].items(), key=lambda x: x[1][0], reverse=True)
        print("attempt: {}, mean of T: initial model {}, output model {}".format(
            attempt, np.mean(initial_model.T[:, attempt, :]), np.mean(model.T[:, attempt, :])))
        print("        top 5 questions (question, [count, avg score]): {}".format(question_count_tuples[:5]))

    print("===step 1: ===")
    # check the relation
    # bias_s vs true student average score
    # bias_a vs true average score over attempts
    # bias_q vs true question average score
    student_score_dict = {}
    attempt_score_dict = {}
    question_score_dict = {}
    print("size of training data {}".format(len(model.train_data)))
    for (student, attempt, question, obs, resource) in model.train_data:
        if resource == 0:
            if student not in student_score_dict: student_score_dict[student] = []
            student_score_dict[student].append(obs)
            if attempt not in attempt_score_dict:
                attempt_score_dict[attempt] = []
            attempt_score_dict[attempt].append(obs)
            if question not in question_score_dict: question_score_dict[question] = []
            question_score_dict[question].append(obs)

    avg_student_score = []
    avg_attempt_score = []
    avg_question_score = []
    for student in range(model.num_users):
        if student not in student_score_dict:
            avg_student_score.append(0.)
        else:
            avg_student_score.append(np.mean(student_score_dict[student]))
    for attempt in range(model.num_attempts):
        if attempt not in attempt_score_dict:
            avg_attempt_score.append(0.)
        else:
            avg_attempt_score.append(np.mean(attempt_score_dict[attempt]))
    for question in range(model.num_questions):
        avg_question_score.append(np.mean(question_score_dict[question]))

    avg_student_score = np.array(avg_student_score)
    avg_attempt_score = np.array(avg_attempt_score)
    avg_question_score = np.array(avg_question_score)

    # analysis of the temporal dynamic
    plt.figure()
    plt.rc('axes', labelsize=18)
    plt.plot(range(model.num_attempts), model.bias_a, 'g-', marker='s', label='attempt bias')
    plt.plot(range(model.num_attempts), np.mean(model.T, axis=(0, 2)), 'b-', marker='X', label='mean of T')
    # plt.plot(range(model.num_attempts), np.mean(true_T[0], axis=(1)), 'c-', marker='*', label='mean of true T[0]')
    # plt.plot(range(model.num_attempts), np.mean(true_T[1], axis=(1)), 'y-', marker='*', label='mean of true T[1]')
    # plt.plot(range(model.num_attempts), np.mean(true_T[2], axis=(1)), 'm-', marker='*', label='mean of true T[2]')

    # plt.plot(range(model.num_attempts), true_T[0,:,0], 'c-', marker='*', label='mean of true T[0,:,0]')
    # plt.plot(range(model.num_attempts), true_T[0,:,1], 'y-', marker='*', label='mean of true T[0,:,1]')
    # plt.plot(range(model.num_attempts), true_T[0,:,2], 'm-', marker='*', label='mean of true T[0,:,2]')


    # plt.plot(range(model.num_attempts), np.mean(T_plus_bias_a, axis=(0, 2)),
    #          'd-', marker='<', label='mean of T + bias_a')
    # plt.plot(range(model.num_attempts), avg_attempt_score, 'r-', marker='^', label='average grade')
    # print(np.mean(model.T, axis=(0, 2)))
    # plt.plot(range(model.num_attempts), np.mean(ST, axis=(0, 2)), 'm-', marker='*', label='mean of ST')
    # plt.plot(range(model.num_attempts), ST_mean, 'm-', marker='*', label='mean of ST')
    print(ST_mean)
    # plt.plot(range(model.num_attempts), np.mean(STQ, axis=(0, 2)), 'd-', marker='>', label='mean of STQ')
    # plt.plot(range(model.num_attempts), avg_attempt_offset, 'k-', marker='*', label='average offset')
    plt.xlabel("index of attempts")
    plt.ylabel("value")
    plt.xticks(np.arange(0, model.num_attempts, 2))
    # plt.title("{} {} Dataset".format(data_str, course_num))
    plt.legend(loc=0)
    plt.savefig("figures/{}/{}/avg_grade_vs_bias_a_{}.png".format(data_str, course_num, fold))
    plt.show()
    plt.clf()

    # plt.figure()
    # plt.plot(range(model.num_users), model.bias_s, 'g-', marker='s', label='student bias')
    # plt.plot(range(model.num_users), avg_student_score, 'r-', marker='^', label='average grade')
    # plt.xlabel("index of students")
    # plt.ylabel("value")
    # plt.xticks(np.arange(0, model.num_users, 10))
    # plt.title("{} {} Dataset".format(data_str, course_num))
    # plt.legend(loc=0)
    # plt.savefig("figures/{}/{}/avg_grade_vs_bias_s_{}.png".format(data_str, course_num, fold))
    # plt.show()
    # plt.clf()

    plt.figure()
    plt.plot(range(model.num_questions), model.bias_q, 'g-', marker='s', label='question bias')
    plt.plot(range(model.num_questions), avg_question_score, 'r-', marker='^', label='average grade')
    plt.xlabel("Index of Question")
    plt.ylabel("Value")
    plt.xticks(np.arange(0, model.num_questions, 2))
    # if data_str == "morf" and course_num == "Quiz_Discussion":
    #     plt.title("MORF_QD")
    # elif data_str == "morf" and course_num == "Quiz_Lecture":
    #     plt.title("MORF_QL")
    # elif data_str == "canvas" and course_num == "770000832960975":
    #     plt.title("CANVAS_H")
    plt.legend(loc=0)
    plt.savefig("figures/{}/{}/avg_score_and_bias_q_correlation.pdf".format(data_str, course_num, fold))
    plt.show()
    plt.clf()
    print("{} spearman correlation  {}".format(course_num, stats.spearmanr(model.bias_q, avg_question_score)))
    print("{} pearson correlation  {}".format(course_num, stats.pearsonr(model.bias_q, avg_question_score)))

    print("=================================================step 3==================================================")
    # average of ST(knowledge) on all students for all concepts
    average_knowledge = []
    for attempt in range(model.num_attempts):
        average_knowledge.append(np.mean(ST[:, attempt, :]))
    plt.figure()
    plt.plot(range(model.num_attempts), average_knowledge, 'r-', marker='^', label='Average Knowledge')
    plt.xlabel("Index of Attempt")
    plt.ylabel("Value")
    plt.xticks(np.arange(0, model.num_attempts, 2))
    # if data_str == "morf" and course_num == "Quiz_Discussion":
    #     plt.title("MORF_QD")
    # elif data_str == "morf" and course_num == "Quiz_Lecture":
    #     plt.title("MORF_QL")
    # elif data_str == "canvas" and course_num == "770000832960975":
    #     plt.title("CANVAS_H")

    plt.legend(loc=0)
    plt.savefig("figures/{}/{}/average_knowledge.pdf".format(data_str, course_num, fold))
    plt.show()
    plt.clf()

    print("=================================================step 4==================================================")
    # average knowledge on all students for each concept
    plt.figure()
    for concept in range(model.num_concepts):
    # for concept in [3,5,7]: # for morf quiz lecture
        average_concept_knowledge = []
        line_mode = concept % 4
        marker_mode = concept % 8
        color_mode = concept % 8
        for attempt in range(model.num_attempts):
            average_concept_knowledge.append(np.mean(ST[:, attempt, concept]))
        # plt.plot(range(model.num_attempts)[5:71], average_concept_knowledge[5:71], marker=marker_dict[marker_mode],
        #          color=color_dict[color_mode], linestyle=line_style_dict[line_mode], label='concept {}'.format(concept))
        plt.plot(range(model.num_attempts), average_concept_knowledge, marker=marker_dict[marker_mode],
                 color=color_dict[color_mode], linestyle=line_style_dict[line_mode], label='concept {}'.format(concept))
    # plt.plot(range(model.num_attempts), average_knowledge, 'k-', marker='*', label='Average Knowledge')
    plt.xlabel("Index of Attempt")
    plt.ylabel("Value")
    plt.xticks(np.arange(0, model.num_attempts, 2), fontsize=15)
    plt.yticks(fontsize=15)
    # plt.xticks(np.arange(5, 71, 5))
    if data_str == "morf" and course_num == "Quiz_Discussion":
        plt.title("MORF_QD", fontsize=18)
    elif data_str == "morf" and course_num == "Quiz_Lecture":
        plt.title("MORF_QL", fontsize=18)
    elif data_str == "canvas" and course_num == "770000832960975":
        plt.title("CANVAS_H", fontsize=18)
    plt.legend(loc='upper right', fontsize=12, ncol=3, columnspacing=0.05)
    # plt.axis((0, 30, 0, 1.0)) # set the limit of axis
    plt.savefig("figures/{}/{}/concept_growth_comparison.pdf".format(data_str, course_num))
    plt.show()
    plt.clf()

    # print("=================================================step 6==================================================")
    # apply K-Means on student matrix
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(model.S)
    # print(kmeans.labels_)
    # for student, label in enumerate(kmeans.labels_):
    #     print(student, label)

    print("=================================================step 5==================================================")
    # # start student modeling
    # # find all students who have tried all attempts, choose two students and
    # # compare their knowledge growth
    active_students = get_active_students(model.train_data, model.num_attempts)
    mat_S = []
    index_student_dict = {}
    for index, student in enumerate(sorted(active_students.keys())):
        # print("index {}, student {}".format(index, student))
        index_student_dict[index] = student
        mat_S.append(model.S[student])
    # TODO: KMeans will do the cluster based on grade, since mat_S is derived based on quiz grade
    # kmeans = KMeans(n_clusters=3, random_state=0).fit(mat_S)
    sp_clusters = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=1).fit(mat_S)
    # computer average grade on each group

    label_cluster_dict = {}
    # for index, label in enumerate(kmeans.labels_):
    for index, label in enumerate(sp_clusters.labels_):
        if label not in label_cluster_dict:
            label_cluster_dict[label] = []
        label_cluster_dict[label].append(index_student_dict[index])
        # print(index, index_student_dict[index], label)

    test_S = []
    t_test = {}
    for label in sorted(label_cluster_dict.keys()):
        print('label {}, size {}, group: {}'.format(label, len(label_cluster_dict[label]), label_cluster_dict[label]))
        avg_cluster_score = []
        t_test[label] = []
        for student in label_cluster_dict[label]:
            test_S.append(model.S[student])
            if student not in student_score_dict:
                pass
            else:
                avg_cluster_score.append(np.mean(student_score_dict[student]))
                t_test[label].append(np.mean(student_score_dict[student]))
        print('average grade on this cluster {}, std {}\n'.format(np.mean(avg_cluster_score), np.std(avg_cluster_score)))

    for label_1 in sorted(label_cluster_dict.keys()):
        for label_2 in sorted(label_cluster_dict.keys()):
            print(label_1, label_2)
            if label_1 == label_2:
                pass
            else:
                vec_a = t_test[label_1]
                vec_b = t_test[label_2]
                print("label {} vs {}, {}".format(label_1, label_2, stats.ttest_ind(vec_a, vec_b)))

    test_S = np.array(test_S)
    print('\n')

    plt.figure()
    count = 0
    for student in sorted(active_students.keys()):
        # if student in [100, 115, 71, 179, 268, 318]:
        # if student in [184, 225, 257, 318]:
        # if student in [257, 318]:
        # TODO for student who has less attempts, we only need to compare the difference up to that attempt
        # if student in [356, 152, 159, 447, 276]:
        # if student in [356, 233]:
        # if student in [0, 2, 11]:
        # if student in [0, 35, 356, 233]:
        # if student in [5, 100]: # for MORF QL
        if student in [0, 3]: # for canvas
        # if student in [0, 8, 189]: # MORF QD
        # if student in [0, 22]:
            # if student != None:
            # if (data_str == 'morf' and course_num == 'Quiz_Discussion' and student in [100, 318]) or
            #     (data_str == 'morf' and course_num == 'Quiz_Lecture' and student in [100, 318]) :
            knowledge = np.mean(ST[student, :, :], axis=1) + model.bias_s[student]  # TODO: do we need bias_s ?
            # knowledge = np.mean(ST[student, :, :], axis=1)   # TODO: do we need bias_s ?
            record = sorted(active_students[student], key=lambda x: x[1])
            print('student {}, record: {}'.format(student, record))
            scores = []
            for (student, attempt, index, obs, resource) in record:
                if resource == 0:
                    scores.append(obs)
            print('student {}, avg quiz score {}, size {},  mean knowledge {} std {}\n'.format(student, np.mean(scores),
                                                                                               len(scores),
                                                                                               np.mean(knowledge),
                                                                                               np.std(knowledge)))
            line_mode = count % 4
            marker_mode = count % 8
            color_mode = count % 8
            count += 1
            # plt.plot(range(model.num_attempts), knowledge, marker=marker_dict[marker_mode],
            #          color=color_dict[color_mode], linestyle=line_style_dict[line_mode],
            #          label='knowledge of student {}'.format(student)) # for morf_QD
            # plt.plot(range(model.num_attempts)[5:71], knowledge[5:71], marker=marker_dict[marker_mode],
            #          color=color_dict[color_mode], linestyle=line_style_dict[line_mode],
            #          label='knowledge of student {}'.format(student)) # For MORF-QL
            plt.plot(range(model.num_attempts)[:-1], knowledge[:-1], marker=marker_dict[marker_mode],
                     color=color_dict[color_mode], linestyle=line_style_dict[line_mode],
                     label='knowledge of student {}'.format(student)) # for canvas-H
    plt.xlabel("Index of Attempt")
    plt.ylabel("Value")
    plt.xticks(np.arange(0, model.num_attempts, 2))
    # plt.xticks(np.arange(5, 71, 5))
    # if data_str == "morf" and course_num == "Quiz_Discussion":
    #     plt.title("MORF_QD")
    # elif data_str == "morf" and course_num == "Quiz_Lecture":
    #     plt.title("MORF_QL")
    # elif data_str == "canvas" and course_num == "770000832960975":
    #     plt.title("CANVAS_H")
    plt.legend(loc=0)
    # plt.axis((0, 6, -1.5, 1.5)) # set the limit of axis
    plt.savefig("figures/{}/{}/knowledges_growth_comparison_over_students.pdf".format(data_str, course_num, fold))
    plt.show()
    plt.clf()

    # print("=================================================step 7==================================================")
    plt.figure()
    count = 0
    for student in sorted(active_students.keys()):
        # if student in [0, 356]:
        if student in [0, 7, 17]:
            for concept in range(model.num_concepts):
                # knowledge = np.mean(ST[student, :, :], axis=1) + model.bias_s[student]  # TODO: do we need bias_s ?
                concept_knowledge = ST[student, :, concept] + model.bias_s[student]
                line_mode = count % 4
                marker_mode = count % 8
                color_mode = count % 8
                count += 1
                plt.plot(range(model.num_attempts), concept_knowledge, marker=marker_dict[marker_mode],
                         color=color_dict[color_mode], linestyle=line_style_dict[line_mode],
                         label='knowledge of student {} on concept {}'.format(student, concept))
    plt.xlabel("index of attempt")
    plt.ylabel("value")
    plt.xticks(np.arange(0, model.num_attempts, 2))
    plt.title("{} {} Dataset: \n knowledge growth on different students and different concept".format(data_str, course_num))
    plt.legend(loc=0)
    # plt.axis((0, 6, -1.5, 1.5)) # set the limit of axis
    # plt.savefig("figures/{}/{}/average_knowledge.png".format(data_str, course_num, fold))
    plt.show()
    plt.clf()
    print("finished.")


def get_active_students(train_data, num_attempts):
    """get all students who have all attempts,
    return active students, and their attempt questions seq.
    """
    active_students = {}
    for (student, attempt, index, obs, resource) in train_data:
        if student not in active_students:
            active_students[student] = []
        active_students[student].append([student, attempt, index, obs, resource])

    # for student in active_students.copy():
        # if len(active_students[student]) != num_attempts:
        # if len(active_students[student]) <= 10:
        #     active_students.pop(student)
    # print("active students {}".format(len(active_students.keys())))
    return active_students


def run_simulation():
    data_str = 'simulation'
    # course_num = 'Quiz_Discussion'
    # course_num = 'Quiz_Discussion_2'
    course_num = 'Quiz_Assignment'
    model_str = 'multiview_obs'

    # step 3: run the single test with best hyperparameters
    skill_dim = 3
    concept_dim = 3
    lambda_s = 0
    lambda_t = 0
    lambda_q = 0
    lambda_e = 0
    lambda_bias = 0
    penalty_weight = 0.1
    markovian = 1
    max_iter = 30
    trade_off = 0.1
    lr = 0.1
    fold = 2
    model_analysis(data_str, course_num, model_str, fold, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q,
                   lambda_e, lambda_bias, lr, penalty_weight, markovian, max_iter, trade_off)


def run_canvas():
    data_str = 'canvas'
    course_num = '770000832960975'
    model_str = 'multiview_obs'

    # step 3: run the single test with best hyperparameters
    # option 1
    # skill_dim = 20
    # concept_dim = 3
    # lambda_s = 0
    # lambda_t = 0
    # lambda_q = 0
    # lambda_e = 0
    # lambda_bias = 0
    # lr = 0.01
    # penalty_weight = 0
    # markovian = 1
    # max_iter = 50
    # trade_off = 0.05
    # fold = 2

    # option 2
    skill_dim = 28
    concept_dim = 7
    lambda_s = 0
    lambda_t = 0
    lambda_q = 0
    lambda_e = 0
    lambda_bias = 0
    lr = 0.01
    penalty_weight = 2.0
    markovian = 0
    max_iter = 50
    trade_off = 0.5
    fold = 2
    model_analysis(data_str, course_num, model_str, fold, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q,
                   lambda_e, lambda_bias, lr, penalty_weight, markovian, max_iter, trade_off)


def run_morf():
    data_str = 'morf'
    model_str = 'multiview_obs'

    course_num = 'Quiz_Discussion'
    # TODO: drop of knowledge at attempt 3 may be caused by some students first did the question 11, then quesiton 9
    # TODO: you can see that the grade from attempt 3 to attempt 4 is decreased
    # TODO: the contrast on question bias is because some questions do not have enough observations
    # step 3: run the single test with best hyperparameters
    # option 1.
    # skill_dim = 20; concept_dim = 9
    # lambda_s = 0; lambda_t = 0.001; lambda_q = 0; lambda_e = 0; lambda_bias = 0
    # lr = 0.1; penalty_weight = 0; markovian = 1; max_iter = 50; trade_off = 0.05
    # option 2.
    # skill_dim = 10; concept_dim = 5
    # lambda_s = 0; lambda_t = 0.005; lambda_q = 0; lambda_e = 0; lambda_bias = 0
    # lr = 0.1; penalty_weight = 0; markovian = 1; max_iter = 50; trade_off = 0.5
    # option 3.
    # skill_dim = 15; concept_dim = 3
    # lambda_s = 0; lambda_t = 0.001; lambda_q = 0; lambda_e = 0; lambda_bias = 0
    # lr = 0.1; penalty_weight = 0.05; markovian = 1; max_iter = 50; trade_off = 1
    # option 4.
    # skill_dim = 15; concept_dim = 5
    # lambda_s = 0; lambda_t = 0.001; lambda_q = 0; lambda_e = 0; lambda_bias = 0
    # lr = 0.1; penalty_weight = 0.1; markovian = 1; max_iter = 50; trade_off = 0.05
    # option 5.

    skill_dim = 39;
    concept_dim = 5
    lambda_s = 0;
    lambda_t = 0;
    lambda_q = 0;
    lambda_e = 0;
    lambda_bias = 0
    lr = 0.1;
    penalty_weight = 0;
    markovian = 0;
    max_iter = 50;
    trade_off = 1.0

    # skill_dim = 10;
    # concept_dim = 4
    # lambda_s = 0;
    # lambda_t = 0.005;
    # lambda_q = 0;
    # lambda_e = 0;
    # lambda_bias = 0
    # lr = 0.1;
    # penalty_weight = 0;
    # markovian = 1;
    # max_iter = 50;
    # trade_off = 0.5

    # course_num = 'Quiz_Lecture'
    # # # step 3: run the single test with best hyperparameters
    # # # option 1.
    # skill_dim = 35; concept_dim = 9
    # lambda_s = 0; lambda_t = 0; lambda_q = 0; lambda_e = 0; lambda_bias = 0
    # lr = 0.1; penalty_weight = 0; markovian = 0; max_iter = 50; trade_off = 0.5

    fold = 4
    model_analysis(data_str, course_num, model_str, fold, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q,
                   lambda_e, lambda_bias, lr, penalty_weight, markovian, max_iter, trade_off)


if __name__ == '__main__':
    run_canvas()
    # run_morf()
    # run_simulation()

    # T = pickle.load(open("data/simulation/Quiz_Discussion_2/simulated_T.pkl", "rb"))

