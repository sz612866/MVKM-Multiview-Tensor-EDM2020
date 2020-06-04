# multiview STQ factorization with only 2 views
# import sys
#
# sys.path.append("../")
from numpy import linalg as LA
from utils import *
import time
import warnings

warnings.filterwarnings("error")


class MultiView(object):
    """
    pure rank based tensor factorization without non-negativity constraint on Q-matrix, and
    the temporal smoothness is enforced on the observed entries of T only.
    """

    def __init__(self, config):
        """
        :param config:
        :var self.exact_penalty: if exact penalty if True, then penalty on T, otherwise on X
        """
        np.random.seed(0)
        self.log_file = config['log_file']
        if self.log_file:
            self.logger = create_logger(self.log_file)
        self.train_data = config['train']
        self.test_users = config['test_users']

        self.num_users = config['num_users']
        self.num_skills = config['num_skills']
        self.num_attempts = config['num_attempts']
        self.num_concepts = config['num_concepts']
        self.num_questions = config['num_questions']
        self.lambda_s = config['lambda_s']
        self.lambda_t = config['lambda_t']
        self.lambda_q = config['lambda_q']
        self.lambda_bias = config['lambda_bias']
        self.penalty_weight = config['penalty_weight']
        self.markovian_steps = config['markovian_steps']
        self.lr = config['lr']
        self.tol = config['tol']
        self.max_iter = config['max_iter']

        self.num_examples = config['num_discussions']
        self.lambda_e = config['lambda_e']
        self.trade_off_e = config['trade_off_example']

        self.binarized_question = True  # apply sigmoid if true
        self.binarized_example = True

        self.exact_penalty = False
        self.log_sigmoid = False
        self.use_bias_a = True  # apply bias term on attempts
        self.current_test_attempt = None
        self.loss_list = []

        self.val_data = []
        self.train_data_markovian = []
        train_data_dict = {}

        for student, attempt, question, obs, resource in self.train_data:
            key = (student, attempt, question, resource)
            if key not in train_data_dict:
                train_data_dict[key] = obs

        train_data_markovian_dict = {}
        for student, attempt, question, obs, resource in self.train_data:
            upper_steps = min(self.num_attempts, attempt + self.markovian_steps + 1)
            for j in range(attempt + 1, upper_steps):
                if (student, j, question, resource) not in train_data_dict:
                    if (student, j, question, resource) not in train_data_markovian_dict:
                        train_data_markovian_dict[(student, j, question, resource)] = True
                        self.train_data_markovian.append((student, j, question, resource))

        self.S = np.random.random_sample((self.num_users, self.num_skills))
        self.T = np.random.random_sample((self.num_skills, self.num_attempts,
                                          self.num_concepts))
        self.Q = np.random.random_sample((self.num_concepts, self.num_questions))
        self.E = np.random.random_sample((self.num_concepts, self.num_examples))
        self.bias_s = np.zeros(self.num_users)
        self.bias_a = np.zeros(self.num_attempts)
        self.bias_q = np.zeros(self.num_questions)
        self.bias_e = np.zeros(self.num_examples)

    def __getstate__(self):
        """
        since the logger cannot be pickled, to avoid the pickle error, we should add this
        :return:
        """
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def _get_question_prediction(self, student, attempt, question):
        """
        predict value at tensor Y[attempt, student, question]
        :param attempt: attempt index
        :param student: student index
        :param question: question index
        :return: predicted value of tensor Y[attempt, student, question]
        """
        pred = np.dot(np.dot(self.S[student, :], self.T[:, attempt, :]), self.Q[:, question])
        if self.use_bias_a:
            pred += self.bias_s[student] + self.bias_a[attempt] + self.bias_q[question]
        else:
            pred += self.bias_s[student] + self.bias_q[question]

        if self.binarized_question:
            pred = sigmoid(pred)
        return pred

    def _get_example_prediction(self, student, attempt, example):
        """
        predict value at tensor Y[attempt, student, question]
        :param attempt: attempt index
        :param student: student index
        :param question: question index
        :return: predicted value of tensor Y[attempt, student, question]
        """
        pred = np.dot(np.dot(self.S[student, :], self.T[:, attempt, :]), self.E[:, example])
        if self.use_bias_a:
            pred += self.bias_s[student] + self.bias_a[attempt] + self.bias_e[example]
        else:
            pred += self.bias_s[student] + self.bias_e[example]

        if self.binarized_example:
            pred = sigmoid(pred)
        return pred

    def _get_loss(self):
        """
        override the function in super class
        compute the loss, which is RMSE of observed records +
        regularization + penalty of temporal non-smoothness
        :return: loss
        """
        loss, square_loss, bias_reg = 0., 0., 0.
        square_loss_q, square_loss_l, square_loss_e = 0., 0., 0.
        q_count, l_count, e_count = 0., 0., 0.
        for (student, attempt, question, obs, resource) in self.train_data:
            if resource == 0:
                pred = self._get_question_prediction(student, attempt, question)
                square_loss_q += (obs - pred) ** 2
                q_count += 1
            elif resource == 1:
                pred = self._get_example_prediction(student, attempt, question)
                square_loss_e += (obs - pred) ** 2
                e_count += 1
        square_loss = square_loss_q + self.trade_off_e * square_loss_e

        reg_S = LA.norm(self.S) ** 2
        reg_T = LA.norm(self.T) ** 2  # regularization on tensor T
        reg_Q = LA.norm(self.Q) ** 2  # regularization on matrix Q
        reg_E = LA.norm(self.E) ** 2

        reg_loss = self.lambda_s * reg_S + self.lambda_q * reg_Q + self.lambda_t * reg_T + \
                   self.lambda_e * reg_E
        loss = square_loss + reg_loss
        q_rmse = np.sqrt(square_loss_q / q_count) if q_count != 0 else 0.
        e_rmse = np.sqrt(self.trade_off_e * square_loss_e / e_count) if e_count != 0 else 0.
        if self.lambda_bias:
            if self.use_bias_a:
                bias_reg = self.lambda_bias * (
                        LA.norm(self.bias_s) ** 2 + LA.norm(self.bias_a) ** 2 +
                        LA.norm(self.bias_q) ** 2 + LA.norm(self.bias_e) ** 2)
            else:
                bias_reg = self.lambda_bias * (
                        LA.norm(self.bias_s) ** 2 + LA.norm(self.bias_q) ** 2 +
                        LA.norm(self.bias_e) ** 2)

        penalty = self._get_penalty()
        loss += bias_reg + penalty
        return loss, q_rmse, e_rmse, reg_loss, penalty, bias_reg

    def _get_penalty(self):
        """
        compute the penalty on the observations, we want all attempts before the obs has smaller
        score, and the score after obs should be greater.
        we use sigmoid to set the penalty between 0 and 1
        if knowledge at current attempt >> prev attempt, then diff is large, that mean
        sigmoid(diff) is large and close to 1., so penalty is a very small negative number
        since we aim to minimize the objective = loss + penalty, the smaller penalty is better

        :return:
        """
        penalty = 0.
        for student, attempt, index, obs, resource in self.train_data:
            if attempt >= 1:
                gap = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                gap[gap > 0.] = 0.
                if self.exact_penalty:
                    diff = np.sum(gap)
                else:
                    if resource == 0:
                        diff = np.dot(np.dot(self.S[student, :], gap), self.Q[:, index])
                    elif resource == 1:
                        diff = np.dot(np.dot(self.S[student, :], gap), self.E[:, index])
                    else:
                        raise AttributeError
                if self.log_sigmoid:
                    diff = np.log(sigmoid(diff))
                penalty -= self.penalty_weight * diff

        for student, attempt, index, resource in self.train_data_markovian:
            if attempt >= 1:
                gap = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                gap[gap > 0.] = 0.
                if self.exact_penalty:
                    diff = np.sum(gap)
                else:
                    if resource == 0:
                        diff = np.dot(np.dot(self.S[student, :], gap), self.Q[:, index])
                    elif resource == 1:
                        diff = np.dot(np.dot(self.S[student, :], gap), self.E[:, index])
                    else:
                        raise AttributeError
                if self.log_sigmoid:
                    diff = np.log(sigmoid(diff))
                penalty -= self.penalty_weight * diff
        return penalty

    def _grad_S_k(self, student, attempt, index, obs=None, resource=None):
        grad = np.zeros_like(self.S[student, :])
        if obs != None:
            if resource == 0:
                pred = self._get_question_prediction(student, attempt, index)
                if self.binarized_question:
                    grad = -2. * (obs - pred) * pred * (1. - pred) * np.dot(self.T[:, attempt, :],
                                                                            self.Q[:, index])
                    if max(grad) > 10.:
                        print("someting wrong")
                else:
                    grad = -2. * (obs - pred) * np.dot(self.T[:, attempt, :], self.Q[:, index])
            elif resource == 1:
                pred = self._get_example_prediction(student, attempt, index)
                if self.binarized_example:
                    grad = -2. * self.trade_off_e * (obs - pred) * pred * (1. - pred) * np.dot(
                        self.T[:, attempt, :], self.E[:, index])
                else:
                    grad = -2. * self.trade_off_e * (obs - pred) * np.dot(self.T[:, attempt, :],
                                                                          self.E[:, index])
        grad += 2. * self.lambda_s * self.S[student, :]

        if not self.exact_penalty:
            if resource == 0:
                if attempt == 0:
                    diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
                elif attempt == self.num_attempts - 1:
                    diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                else:
                    diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                    diff += self.T[:, attempt + 1, :] - self.T[:, attempt, :]
                diff[diff > 0.] = 0.
                val = np.dot(diff, self.Q[:, index])
                grad -= self.penalty_weight * (-1.) * val
            elif resource == 1:
                if attempt == 0:
                    diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
                elif attempt == self.num_attempts - 1:
                    diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                else:
                    diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                    diff += self.T[:, attempt + 1, :] - self.T[:, attempt, :]
                diff[diff > 0.] = 0.
                val = np.dot(diff, self.E[:, index])
                grad -= self.penalty_weight * (-1.) * val
        return grad

    def _grad_T_ij(self, student, attempt, index, obs=None, resource=None):
        """
        compute the gradient of loss w.r.t a specific student j's knowledge at
        a specific attempt i: T_{i,j,:},
        :param attempt: index
        :param student: index
        :param question: index
        :param obs: observation
        :return:
        """

        grad = np.zeros_like(self.T[:, attempt, :])
        if obs != None:
            if resource == 0:
                pred = self._get_question_prediction(student, attempt, index)
                if self.binarized_question:
                    grad = -2. * (obs - pred) * pred * (1. - pred) * np.outer(self.S[student, :],
                                                                              self.Q[:, index])
                else:
                    grad = -2. * (obs - pred) * np.outer(self.S[student, :], self.Q[:, index])
            elif resource == 1:
                pred = self._get_example_prediction(student, attempt, index)
                if self.binarized_example:
                    grad = -2. * self.trade_off_e * (obs - pred) * pred * (1. - pred) * np.outer(
                        self.S[student, :],
                        self.E[:, index])
                else:
                    grad = -2. * self.trade_off_e * (obs - pred) * np.outer(self.S[student, :],
                                                                            self.E[:, index])
        grad += 2. * self.lambda_t * self.T[:, attempt, :]

        if resource == 0:
            if attempt == 0:
                diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
                diff[diff > 0.] = 0.
                if self.exact_penalty:
                    diff[diff < 0.] = -1.
                    grad -= self.penalty_weight * diff
                else:
                    diff = np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    grad -= self.penalty_weight * diff * np.outer(self.S[student, :],
                                                                  self.Q[:, index])
            elif attempt == self.num_attempts - 1:
                diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                diff[diff > 0.] = 0.
                if self.exact_penalty:
                    diff[diff < 0.] = 1.
                    grad -= self.penalty_weight * diff
                else:
                    diff = np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    grad -= self.penalty_weight * diff * (-1.) * np.outer(self.S[student, :],
                                                                          self.Q[:, index])
            else:
                diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                diff[diff > 0.] = 0.
                if self.exact_penalty:
                    diff[diff < 0.] = 1.
                    grad -= self.penalty_weight * diff
                else:
                    diff = np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    grad -= self.penalty_weight * diff * (-1.) * np.outer(self.S[student, :],
                                                                          self.Q[:, index])

                diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
                diff[diff > 0.] = 0.
                if self.exact_penalty:
                    diff[diff < 0.] = -1.
                    grad -= self.penalty_weight * diff
                else:
                    diff = np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    grad -= self.penalty_weight * diff * np.outer(self.S[student, :],
                                                                  self.Q[:, index])
        elif resource == 1:
            if attempt == 0:
                diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
                diff[diff > 0.] = 0.
                if self.exact_penalty:
                    diff[diff < 0.] = -1.
                    grad -= self.penalty_weight * diff
                else:
                    diff = np.dot(np.dot(self.S[student, :], diff), self.E[:, index])
                    grad -= self.penalty_weight * diff * np.outer(self.S[student, :],
                                                                  self.E[:, index])
            elif attempt == self.num_attempts - 1:
                diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                diff[diff > 0.] = 0.
                if self.exact_penalty:
                    diff[diff < 0.] = 1.
                    grad -= self.penalty_weight * diff
                else:
                    diff = np.dot(np.dot(self.S[student, :], diff), self.E[:, index])
                    grad -= self.penalty_weight * diff * (-1.) * np.outer(self.S[student, :],
                                                                          self.E[:, index])
            else:
                diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                diff[diff > 0.] = 0.
                if self.exact_penalty:
                    diff[diff < 0.] = 1.
                    grad -= self.penalty_weight * diff
                else:
                    diff = np.dot(np.dot(self.S[student, :], diff), self.E[:, index])
                    grad -= self.penalty_weight * diff * (-1.) * np.outer(self.S[student, :],
                                                                          self.E[:, index])

                diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
                diff[diff > 0.] = 0.
                if self.exact_penalty:
                    diff[diff < 0.] = -1.
                    grad -= self.penalty_weight * diff
                else:
                    diff = np.dot(np.dot(self.S[student, :], diff), self.E[:, index])
                    grad -= self.penalty_weight * diff * np.outer(self.S[student, :],
                                                                  self.E[:, index])
        return grad

    def _grad_Q_k(self, student, attempt, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific concept-question association
        of a question in Q-matrix,
        :param attempt: index
        :param student:  index
        :param question:  index
        :param obs: the value at Y[attempt, student, question]
        :return:
        """
        grad = np.zeros_like(self.Q[:, question])
        if obs != None:
            pred = self._get_question_prediction(student, attempt, question)
            if self.binarized_question:
                grad = -2. * (obs - pred) * pred * (1. - pred) * np.dot(self.S[student, :],
                                                                        self.T[:, attempt, :])
            else:
                grad = -2. * (obs - pred) * np.dot(self.S[student, :], self.T[:, attempt, :])
        grad += 2. * self.lambda_q * self.Q[:, question]

        if not self.exact_penalty:
            if attempt == 0:
                diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
            elif attempt == self.num_attempts - 1:
                diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
            else:
                diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                diff += self.T[:, attempt + 1, :] - self.T[:, attempt, :]
            diff[diff > 0.] = 0.
            val = np.dot(self.S[student, :], diff)
            grad -= self.penalty_weight * (-1.) * val
        return grad

    def _grad_E_k(self, student, attempt, example, obs=None):
        """
        compute the gradient of loss w.r.t a specific concept-question association
        of a question in Q-matrix,
        :param attempt: index
        :param student:  index
        :param question:  index
        :param obs: the value at Y[attempt, student, question]
        :return:
        """
        grad = np.zeros_like(self.E[:, example])
        if obs != None:
            pred = self._get_example_prediction(student, attempt, example)
            if self.binarized_example:
                grad = -2. * self.trade_off_e * (obs - pred) * pred * (1. - pred) * np.dot(
                    self.S[student, :], self.T[:, attempt, :])
            else:
                grad = -2. * self.trade_off_e * (obs - pred) * np.dot(self.S[student, :],
                                                                      self.T[:, attempt, :])
        grad += 2. * self.lambda_e * self.E[:, example]

        if not self.exact_penalty:
            if attempt == 0:
                diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
            elif attempt == self.num_attempts - 1:
                diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
            else:
                diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                diff += self.T[:, attempt + 1, :] - self.T[:, attempt, :]
            diff[diff > 0.] = 0.
            val = np.dot(self.S[student, :], diff)
            grad -= self.penalty_weight * (-1.) * val
        return grad

    def _grad_bias_s(self, student, attempt, index, obs=None, resource=None):
        """
        compute the gradient of loss w.r.t a specific bias_s
        :param attempt:
        :param student:
        :param question:
        :param obs:
        :return:
        """
        grad = 0.
        if obs != None:
            if resource == 0:
                pred = self._get_question_prediction(student, attempt, index)
                if self.binarized_question:
                    grad -= 2. * (obs - pred) * pred * (1. - pred)
                else:
                    grad -= 2. * (obs - pred)
            elif resource == 1:
                pred = self._get_example_prediction(student, attempt, index)
                if self.binarized_example:
                    grad -= 2. * self.trade_off_e * (obs - pred) * pred * (1. - pred)
                else:
                    grad -= 2. * self.trade_off_e * (obs - pred)
        grad += 2.0 * self.lambda_bias * self.bias_s[student]
        return grad

    def _grad_bias_a(self, student, attempt, index, obs=None, resource=None):
        """
        compute the gradient of loss w.r.t a specific bias_a
        :param attempt:
        :param student:
        :param question:
        :param result:
        :return:
        """
        grad = 0.
        if obs != None:
            if resource == 0:
                pred = self._get_question_prediction(student, attempt, index)
                if self.binarized_question:
                    grad -= 2. * (obs - pred) * pred * (1. - pred)
                else:
                    grad -= 2. * (obs - pred)
            elif resource == 1:
                pred = self._get_example_prediction(student, attempt, index)
                if self.binarized_example:
                    grad -= 2. * self.trade_off_e * (obs - pred) * pred * (1. - pred)
                else:
                    grad -= 2. * self.trade_off_e * (obs - pred)
        grad += 2.0 * self.lambda_bias * self.bias_a[attempt]
        return grad

    def _grad_bias_q(self, student, attempt, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_q
        :param attempt:
        :param student:
        :param question:
        :param obs:
        :return:
        """
        grad = 0.
        if obs != None:
            pred = self._get_question_prediction(student, attempt, question)
            if self.binarized_question:
                grad -= 2. * (obs - pred) * pred * (1. - pred)
            else:
                grad -= 2. * (obs - pred)
        grad += 2. * self.lambda_bias * self.bias_q[question]
        return grad

    def _grad_bias_e(self, student, attempt, example, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_q
        :param attempt:
        :param student:
        :param question:
        :param obs:
        :return:
        """
        grad = 0.
        if obs != None:
            pred = self._get_example_prediction(student, attempt, example)
            if self.binarized_example:
                grad -= 2. * self.trade_off_e * (obs - pred) * pred * (1. - pred)
            else:
                grad -= 2. * self.trade_off_e * (obs - pred)
        grad += 2. * self.lambda_bias * self.bias_e[example]
        return grad

    def _optimize_sgd(self, student, attempt, index, obs=None, resource=None):
        """
        train the T and Q with stochastic gradient descent
        :param attempt:
        :param student:
        :param question:
        :return:
        """

        if resource == 0:
            # optimize Q
            grad_q = self._grad_Q_k(student, attempt, index, obs)
            self.Q[:, index] -= self.lr * grad_q
            self.Q[:, index][self.Q[:, index] < 0.] = 0.
            if self.lambda_q == 0.:
                sum = np.sum(self.Q[:, index])
                if sum != 0:
                    self.Q[:, index] /= sum  # normalization
        elif resource == 1:
            # optimize E
            grad_e = self._grad_E_k(student, attempt, index, obs)
            self.E[:, index] -= self.lr * grad_e
            self.E[:, index][self.E[:, index] < 0.] = 0.
            if self.lambda_e == 0.:
                sum = np.sum(self.E[:, index])
                if sum != 0:
                    self.E[:, index] /= sum  # normalization

        # optimize S
        # training_start_time = time.time()
        grad_s = self._grad_S_k(student, attempt, index, obs, resource)
        self.S[student, :] -= self.lr * grad_s
        if self.lambda_s == 0.:
            self.S[student, :][self.S[student, :] < 0.] = 0.
            sum = np.sum(self.S[student, :])
            if sum != 0:
                self.S[student, :] /= sum

        # the updated Q will be used for computing gradient of T
        grad_t = self._grad_T_ij(student, attempt, index, obs, resource)
        self.T[:, attempt, :] -= self.lr * grad_t

        # train the bias(es)
        if resource == 0:
            self.bias_q[index] -= self.lr * self._grad_bias_q(student, attempt, index, obs)
        elif resource == 1:
            self.bias_e[index] -= self.lr * self._grad_bias_e(student, attempt, index, obs)

        self.bias_s[student] -= self.lr * self._grad_bias_s(student, attempt, index, obs, resource)
        if self.use_bias_a:
            self.bias_a[attempt] -= self.lr * self._grad_bias_a(student, attempt, index, obs,
                                                                resource)

    def training(self):
        """
        minimize the loss until converged or reach the maximum iterations
        with stochastic gradient descent
        :param lr: learning rate
        :param tol:
        :param max_iter: maximum number of iteration for training
        :return:
        """
        # generate validation data for checking stopping criterior
        train_question = []
        for student, attempt, question, obs, resource in self.train_data:
            if resource == 0:
                train_question.append((student, attempt, question, obs, resource))

        np.random.shuffle(train_question)
        self.val_data = train_question[:int(len(train_question) * 0.2)]
        train_question_size = len(train_question) - len(self.val_data)
        for record in self.val_data:
            try:
                self.train_data.remove(record)
            except:
                print(record)
                print(record in self.val_data)
                print(record in self.train_data)
                print(sorted(self.train_data, key=lambda x:x[1]))
        self.logger.info(strRed('test attempt: {}, train size: {}, val size {}'.format(
            self.current_test_attempt, len(self.train_data), len(self.val_data))))

        loss, q_rmse, e_rmse, reg_loss, penalty, bias_reg = self._get_loss()
        self.logger.info("initial: lr: {:.4f}, loss: {:.2f}, q_rmse: {:.5f}, reg_T: {:.2f}, "
                         "penalty: {:.5f}, bias_reg: {:.3f}".format(self.lr, loss, q_rmse, reg_loss,
                                                                    penalty, bias_reg))
        self.loss_list.append(loss)

        self.logger.info(strBlue("*" * 40 + "[ Training Results ]" + "*" * 40))
        iter = 0
        train_perf = []
        val_perf = []
        val_q_rmse_list = [1.]
        start_time = time.time()
        converge = False
        min_iters = 10
        while not converge:
            np.random.shuffle(self.train_data)
            np.random.shuffle(self.train_data_markovian)
            best_S = np.copy(self.S)
            best_T = np.copy(self.T)
            best_Q = np.copy(self.Q)
            best_E = np.copy(self.E)
            best_bias_s = np.copy(self.bias_s)
            best_bias_a = np.copy(self.bias_a)
            best_bias_q = np.copy(self.bias_q)
            best_bias_e = np.copy(self.bias_e)

            for (student, attempt, index, obs, resource) in self.train_data:
                self._optimize_sgd(student, attempt, index, obs, resource=resource)

            for (student, attempt, index, resource) in self.train_data_markovian:
                self._optimize_sgd(student, attempt, index, resource=resource)

            preds = np.zeros((self.num_users, self.num_attempts, self.num_questions))
            for att in range(self.num_attempts):
                preds[:, att, :] = np.dot(np.dot(self.S, self.T[:, att, :]), self.Q)

            loss, q_rmse, e_rmse, reg_loss, penalty, bias_reg = self._get_loss()
            val_q_count, val_q_rmse, _, = self.testing(self.val_data, validation=True)
            train_perf.append([train_question_size, q_rmse])
            val_perf.append([val_q_count, val_q_rmse])

            run_time = time.time() - start_time
            self.logger.info("iter: {}, lr: {:.4f}, total loss: {:.2f}, reg_T: {:.2f}, penalty: "
                             "{:.5f}, bias_reg: {:.3f}, run time: {:.2f}".format(
                iter, self.lr, loss, reg_loss, penalty, bias_reg, run_time))
            self.logger.info("Train:  train_q_count: {}, train_q_rmse: {:.5f}".format(
                train_question_size, q_rmse))
            self.logger.info("Validation: val_q_count: {}, val_q_rmse: {:.5f}".format(
                val_q_count, val_q_rmse))

            if iter == self.max_iter:
                self.logger.info("=" * 50)
                self.logger.info("** converged **, condition: 0, iter: {}".format(iter))
                self.loss_list.append(loss)
                converge = True
            elif iter >= min_iters and abs(val_q_rmse - val_q_rmse_list[-1]) < self.tol:
                self.logger.info("=" * 50)
                self.logger.info("** converged **, condition: 1, iter: {}".format(iter))
                self.loss_list.append(loss)
                converge = True
            elif iter >= min_iters and val_q_rmse >= np.mean(
                    val_q_rmse_list[-3:]):  # early stopping
                self.logger.info("=" * 40)
                self.logger.info("** converged **, condition: 2, iter: {}".format(iter))
                converge = True
            elif iter >= min_iters and loss >= np.mean(self.loss_list[-3:]):  # early stopping
                self.logger.info("=" * 40)
                self.logger.info("** converged **, condition: 3, iter: {}".format(iter))
                converge = True
            elif val_q_rmse >= val_q_rmse_list[-1]:
                self.loss_list.append(loss)
                val_q_rmse_list.append(val_q_rmse)
                iter += 1
                self.lr *= 0.5
            elif loss == np.nan:
                self.lr *= 0.1
            else:
                self.loss_list.append(loss)
                val_q_rmse_list.append(val_q_rmse)
                iter += 1

        # reset to previous S, T, Q
        self.S = best_S
        self.T = best_T
        self.Q = best_Q
        self.E = best_E
        self.bias_s = best_bias_s
        self.bias_a = best_bias_a
        self.bias_q = best_bias_q
        self.bias_e = best_bias_e
        for record in self.val_data:
            self.train_data.append(record)

        return train_perf[-1], val_perf[-1]

    def testing(self, test_data, validation=False):
        """
        for 5-fold cross validation, 4 fold are used for training, and the rest is used for testing.
        :return: performance metrics mean squared error, RMSE, and mean absolute error
        """
        # check current temporal difference on training data, then apply the diff to testing data
        # temporal_diff = []
        # for (attempt, student, question, obs) in self.train_data:
        #     if attempt == self.current_test_attempt:
        #         diff = self.T[attempt, student, :] - self.T[attempt-1, student, :]
        #         temporal_diff.append(diff)
        # mean_diff = np.mean(temporal_diff)

        q_count, square_error, abs_error = [0.] * 3
        if not validation:
            self.logger.info(strBlue("*" * 40 + "[ Testing Results ]" + "*" * 40))
            self.logger.info("current testing attempt: {}, test size: {}".format(
                self.current_test_attempt, len(test_data)))

        for (student, attempt, question, obs, resource) in test_data:
            if resource == 0:
                q_count += 1.
                pred = self._get_question_prediction(student, attempt, question)
                square_error += (obs - pred) ** 2
                abs_error += abs(obs - pred)
        if q_count == 0:
            return [0.] * 3
        else:
            rmse = np.sqrt(square_error / q_count)
            mae = abs_error / q_count
            if not validation:
                self.logger.info("Test Size: {}, RMSE: {:.5f}, MAE: {:.5f}\n".format(
                    q_count, rmse, mae))
            return [q_count, rmse, mae]
