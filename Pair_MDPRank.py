import json
import datetime
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ' '

import numpy as np
import tensorflow as tf

n_memory = 8
n_batch = 4
n_sample = 16
n_copy = 4
top_n = 10

DATASET = 'MSLRWeb30K'
FOLD = 'Fold1'

MODE = 'pair'
# MODE = 'mdprank'

print(DATASET, FOLD)

LR = 0.001
FEATURE_DIM = 45

date_time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
print(date_time)

file_folder = '../log/MDPRank/' + DATASET + '/' + MODE + '/' + FOLD + '/' + date_time + '/'

os.makedirs(file_folder)
model_file = file_folder + '/Model'
result_file = open(file_folder + '/result_file', 'w')

# Dir = '../data/'
Dir = '/home/xu/zw/data/'

QUERY_DOC = json.load(open(Dir + DATASET + '_' + FOLD + '_query_doc', 'r'))
DOC_REPR = json.load(open(Dir + DATASET + '_' + FOLD + '_doc_feature', 'r'))
QUERY_TEST = json.load(open(Dir + DATASET + '_' + FOLD + '_test_query', 'r'))
QUERY_TRAIN = json.load(open(Dir + DATASET + '_' + FOLD + '_train_query', 'r'))
QUERY_VAL = json.load(open(Dir + DATASET + '_' + FOLD + '_vali_query', 'r'))
QUERY_DOC_TRUTH = json.load(open(Dir + DATASET + '_' + FOLD + '_query_doc_label', 'r'))
MAX_DCG = json.load(open(Dir + DATASET + '_' + FOLD + '_best_dcg', 'r'))
print(len(QUERY_TRAIN))

hsNdcgRelScore = [0, 1, 3, 7, 15]  # {'4':15,'3':7,'2':3,'1':1,'0':0}
hsPrecisionRel = [0, 1, 1, 1, 1]  # {'4':1,'3':1,'2':1,'1':1,'0':0}


class Ranker(object):
    def __init__(self):
        self.action_set = tf.placeholder(tf.float32, [None, FEATURE_DIM], name='action_set')
        self.label_set = tf.placeholder(tf.float32, [None], name='label_set')
        self.action_id_set = tf.placeholder(tf.int32, [None], name='action_id_set')
        self.pos_set = tf.placeholder(tf.int32, name='pos_set')

        self.w = tf.Variable(tf.random_normal([FEATURE_DIM, 1], stddev=0.0001))
        self.score_all = tf.reshape(tf.matmul(self.action_set, self.w), [-1])

        self.action_id_result = tf.distributions.Categorical(probs=tf.nn.softmax(self.score_all)).sample(n_sample)

        n_iter = tf.constant(0, dtype=tf.int32)
        rewards = tf.fill([n_sample], 0.0)
        n_doc = tf.shape(self.action_set)[0]

        multiplier = tf.constant(
            [1, 1, np.log(2) / np.log(3), np.log(2) / np.log(4), np.log(2) / np.log(5), np.log(2) / np.log(6),
             np.log(2) / np.log(7), np.log(2) / np.log(8), np.log(2) / np.log(9), np.log(2) / np.log(10)],
            dtype=tf.float32)

        def cond_fun_rank(iters, rank_list, label_list):
            len_list = tf.cond(tf.less(top_n - self.pos_set - 1, n_doc - 1), lambda: top_n - self.pos_set - 1,
                               lambda: n_doc - 1)
            return tf.less(iters, len_list)

        def loop_fun_rank(iters, rank_list, label_list):
            score = tf.reshape(tf.matmul(tf.reshape(rank_list[:, iters:, :], [-1, FEATURE_DIM]), self.w),
                               [n_copy, -1])
            policy = tf.nn.softmax(score)
            dist = tf.reshape(tf.distributions.Categorical(probs=policy).sample(1) + n_iter, [-1])

            bias = tf.fill([n_copy], iters)

            idx_1 = tf.stack([tf.range(n_copy), tf.add(dist, bias)], axis=1)
            idx_2 = tf.stack([tf.range(n_copy), bias], axis=1)

            rank_list_update = tf.gather_nd(rank_list, idx_1) - tf.gather_nd(rank_list, idx_2)
            label_list_update = tf.gather_nd(label_list, idx_1) - tf.gather_nd(label_list, idx_2)

            rank_list_add = tf.scatter_nd(idx_1, rank_list_update, tf.shape(rank_list))
            rank_list_sub = tf.scatter_nd(idx_2, rank_list_update, tf.shape(rank_list))
            rank_list = rank_list - rank_list_add + rank_list_sub

            label_list_add = tf.scatter_nd(idx_1, label_list_update, tf.shape(label_list))
            label_list_sub = tf.scatter_nd(idx_2, label_list_update, tf.shape(label_list))
            label_list = label_list - label_list_add + label_list_sub

            iters = tf.add(iters, 1)
            return iters, rank_list, label_list

        def cond_fun(iters, reward):
            return tf.less(iters, n_sample)

        def loop_fun(iters, reward):
            idx_1 = [[0]]
            idx_2 = [[self.action_id_result[iters]]]

            rank_list_update = tf.gather_nd(self.action_set, idx_1) - tf.gather_nd(self.action_set, idx_2)
            label_list_update = tf.gather_nd(self.label_set, idx_1) - tf.gather_nd(self.label_set, idx_2)

            rank_list_add = tf.scatter_nd(idx_1, rank_list_update, tf.shape(self.action_set))
            rank_list_sub = tf.scatter_nd(idx_2, rank_list_update, tf.shape(self.action_set))
            rank_list = self.action_set - rank_list_add + rank_list_sub

            label_list_add = tf.scatter_nd(idx_1, label_list_update, tf.shape(self.label_set))
            label_list_sub = tf.scatter_nd(idx_2, label_list_update, tf.shape(self.label_set))
            label_list = self.label_set - label_list_add + label_list_sub

            reward_update = label_list[0] * multiplier[0 + self.pos_set]

            _, _, sample_reward = tf.while_loop(cond_fun_rank, loop_fun_rank,
                                                loop_vars=[n_iter,
                                                           tf.tile(tf.expand_dims(rank_list[1:, :], axis=0),
                                                                   [n_copy, 1, 1]),
                                                           tf.tile(tf.expand_dims(label_list[1:], axis=0),
                                                                   [n_copy, 1])])

            len_list = tf.cond(tf.less(top_n - self.pos_set, n_doc), lambda: top_n - self.pos_set, lambda: n_doc)
            sample_reward = tf.multiply(sample_reward[:, :len_list - 1],
                                        tf.tile(tf.expand_dims(multiplier[self.pos_set + 1:self.pos_set + len_list],
                                                               axis=0),
                                                [n_copy, 1]))
            sample_reward = tf.reduce_mean(tf.reduce_sum(sample_reward, axis=1), axis=0)

            reward_update = reward_update + sample_reward
            reward = reward + tf.scatter_nd([[iters]], [reward_update], tf.shape(reward))

            iters = tf.add(iters, 1)

            return iters, reward

        _, self.reward_result = tf.while_loop(cond_fun, loop_fun, loop_vars=[n_iter, rewards])

        self.loss = tf.reduce_sum(tf.multiply(self.label_set,
                                              tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                  logits=tf.tile(tf.expand_dims(self.score_all, axis=0),
                                                                 [n_sample, 1]),
                                                  labels=self.action_id_set)))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)
            self.gradient = tf.train.AdamOptimizer(LR).compute_gradients(self.loss)
            # tvars = tf.trainable_variables()
            self.grads_set = [(tf.placeholder(dtype=tf.float32, shape=g.get_shape()), v) for (g, v) in self.gradient]
            self.update = tf.train.AdamOptimizer(LR).apply_gradients(self.grads_set)

        gpuConfig = tf.ConfigProto(allow_soft_placement=True)
        gpuConfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpuConfig)
        self.sess.run(tf.global_variables_initializer())
        with tf.variable_scope('saver'):
            self.saver = tf.train.Saver(max_to_keep=0)
            self.saver.save(self.sess, model_file)

    def interaction(self, action, label, pos):
        return self.sess.run([self.action_id_result, self.reward_result],
                             {self.action_set: action, self.label_set: label, self.pos_set: pos})

    def get_score(self, action):
        return self.sess.run(self.score_all, {self.action_set: action})

    def update_trajectories(self, trajectories):
        print('******************* begin update_trajectories ******************************')
        for data in trajectories.values():
            for pos in data.keys():
                action = data[pos]['action']
                reward = data[pos]['reward']
                action_id = data[pos]['action_id']
                for idx in range(n_memory):
                    self.sess.run(self.train_op, {self.action_set: action[idx],
                                                  self.pos_set: pos,
                                                  self.action_id_set: action_id[idx],
                                                  self.label_set: reward[idx]})

    def save_model(self, ite):
        self.saver.save(self.sess, model_file, global_step=ite, write_meta_graph=False)

    def update_trajectories_v1(self, trajectories):
        grads = []
        for data in trajectories.values():
            for pos in data.keys():
                action = data[pos]['action']
                reward = data[pos]['reward']
                action_id = data[pos]['action_id']
                for idx in range(n_memory):
                    grads.append(self.sess.run(self.gradient, {self.action_set: action[idx],
                                                               self.pos_set: pos,
                                                               self.action_id_set: action_id[idx],
                                                               self.label_set: reward[idx]}))

        grads_sum = {}
        for i in range(len(self.grads_set)):
            k = self.grads_set[i][0]
            grads_sum[k] = sum([g[i][0] for g in grads]) / len(grads)

        self.sess.run(self.update, feed_dict=grads_sum)

class Reward(object):

    def reward_ranklist(self, rate_list, s_pos=0):
        reward_list = []
        for r_pos, rate in enumerate(rate_list):
            pos = r_pos + s_pos
            if pos < 2:
                reward_list.append(rate)
            else:
                reward_list.append(round(rate * np.log(2.0) / np.log(pos + 1.0), 6))

        for doc_pos in range(len(reward_list) - 1):
            reward_list[doc_pos + 1] += reward_list[doc_pos]
        return reward_list

    def ndcg_ranklist(self, query, rate_list):

        n_doc = min(len(rate_list), top_n)
        dcg_list = self.reward_ranklist(rate_list[:n_doc])

        if n_doc < top_n:
            dcg_list += (top_n - n_doc) * [0.0]

        for doc_pos in range(top_n):
            if MAX_DCG[query][doc_pos] > 0:
                dcg_list[doc_pos] = dcg_list[doc_pos] / MAX_DCG[query][doc_pos]

        return dcg_list


class SearchEngine(object):
    def __init__(self):
        self.model = Ranker()
        self.reward_model = Reward()
        self.trajectories = {}
        self.data = {}

        print("******  init data *********")
        self.init_data()
        self.init_trajectory()

        print("******  end data *********")


    def init_data(self):
        for query in QUERY_DOC.keys():
            self.data[query] = {'feature': [], 'label': []}
            for doc in QUERY_DOC[query]:
                self.data[query]['feature'].append(DOC_REPR[doc])
                if doc in QUERY_DOC_TRUTH[query].keys():
                    self.data[query]['label'].append(hsNdcgRelScore[QUERY_DOC_TRUTH[query][doc]])
                else:
                    self.data[query]['label'].append(0)

    def init_trajectory(self):
        for query in QUERY_TRAIN:
            self.trajectories[query] = {}
            n_doc = len(QUERY_DOC[query])
            len_list = min(n_doc, top_n)

            for pos in range(len_list):
                self.trajectories[query][pos] = {'action': [], 'action_id': [], 'reward': []}

            for _ in range(n_memory):
                feature = self.data[query]['feature']
                label = self.data[query]['label']

                for pos in range(len_list):
                    action_id, reward = self.model.interaction(feature, label, pos)

                    self.trajectories[query][pos]['action'].append(feature)
                    self.trajectories[query][pos]['action_id'].append(action_id)

                    if MODE == 'pair':
                        reward = reward - np.mean(reward)

                    self.trajectories[query][pos]['reward'].append(reward)

                    idx = action_id[np.argmax(reward)]
                    feature = np.delete(feature, idx, axis=0)
                    label = np.delete(label, idx, axis=0)

    def gen_trajectory(self):
        print('******************* begin gen_trajectory ******************************')
        reward_sum = 0
        for query in QUERY_TRAIN:
            permutation = np.random.permutation(n_memory)

            for idx_pos in range(n_batch):
                feature = self.data[query]['feature']
                label = self.data[query]['label']

                idx = permutation[idx_pos]

                for pos in self.trajectories[query].keys():
                    action_id, reward = self.model.interaction(feature, label, pos)
                    self.trajectories[query][pos]['action'][idx] = feature
                    self.trajectories[query][pos]['action_id'][idx] = action_id

                    reward_sum += np.sum(reward)
                    if MODE == 'pair':
                        reward = reward - np.mean(reward)

                    self.trajectories[query][pos]['reward'][idx] = reward

                    idx_max = action_id[np.argmax(reward)]
                    feature = np.delete(feature, idx_max, axis=0)
                    label = np.delete(label, idx_max, axis=0)
        print('reward_sum', reward_sum)
        run_result = {'reward_sum': reward_sum}
        result_file.write(json.dumps(run_result) + '\n')

    # ***************************************  Test  *******************************************#

    def online_rank(self, query):
        action = self.data[query]['feature']
        label = self.data[query]['label']

        score = self.model.get_score(action)
        pos_idx = np.argsort(-score)
        lens = min(len(pos_idx), 10)

        return [label[idx] for idx in pos_idx[:lens]]

    def test_model(self, query_list=QUERY_TEST, key='test'):
        measure_list = []
        for query in query_list:
            label_list = self.online_rank(query)
            measure_list.append(self.reward_model.ndcg_ranklist(query, label_list))

        result = np.mean(np.asarray(measure_list), 0).tolist()
        run_result = {'key': key, 'ndcg': result}
        result_file.write(json.dumps(run_result) + '\n')
        print(key, result[0], result[4], result[9])

    # ***************************************  Train  *******************************************#

    def train(self):
        self.test_model(QUERY_TEST, 'test')
        self.test_model(QUERY_TRAIN, 'train')
        self.test_model(QUERY_VAL, 'vali')

        for ite in range(20000):
            self.gen_trajectory()
            self.model.update_trajectories(self.trajectories)

            if ite % 1 == 0:
                print(ite)
                self.model.save_model(ite)
                self.test_model(QUERY_TEST, 'test')
                self.test_model(QUERY_TRAIN, 'train')
                self.test_model(QUERY_VAL, 'vali')


SE_model = SearchEngine()
SE_model.train()
