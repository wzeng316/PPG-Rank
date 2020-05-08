import json
import datetime
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from collections import Counter

import numpy as np
import tensorflow as tf

n_memory = 64
N_BATCH = 32
n_sample = 16
n_copy = 8
top_n = 10

FOLD = 'folder5'
print FOLD

LR = 0.0001
FEATURE_DIM = 100
HIDDEN_DIM = 10

date_time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
print date_time

file_folder = '../log/' + date_time
os.makedirs(file_folder)
model_file = file_folder + '/Model'
result_file = open(file_folder + '/result_file', 'w')

Dir = '/home/zengwei/RL4IR/data/DIV/'
QUERY_DOC_TRUTH = json.load(open(Dir + 'query_doc_truth.json', 'r'))
MAX_DCG = json.load(open(Dir + 'max_DCG.json', 'r'))

QUERY_DOC = json.load(open(Dir + 'query_doc.json', 'r'))
DOC_REPR = json.load(open(Dir + 'doc_representation.dat', 'r'))
QUERY_REPR = json.load(open(Dir + 'query_representation.dat', 'r'))
QUERY_ALL = json.load(open(Dir + 'query_' + FOLD + '.json', 'r'))
QUERY_TEST = QUERY_ALL['test_query']
QUERY_TRAIN = QUERY_ALL['train_query'] + QUERY_ALL['vali_query']


class Ranker(object):
    def query_net(self, query):
        # query => 1 * FEATURE_DIM
        # output :
        # query_reps => 1 * HIDDEN_DIM
        with tf.name_scope('query_net'):
            query_reps = tf.layers.dense(query, HIDDEN_DIM, tf.sigmoid, reuse=tf.AUTO_REUSE, name='query_net_2')
        return query_reps

    def state_net(self, ranked_docs, initial_state):
        # ranked_docs => n_batch * n_ranked_docs * FEATURE_DIM
        # initial_state => n_batch * HIDDEN_DIM
        # output :
        # doc_repr => n_batch * HIDDEN_DIM
        with tf.name_scope('state_net'):
            _, doc_repr = tf.nn.dynamic_rnn(dtype=tf.float32, cell=self.rnn_cell,
                                            inputs=ranked_docs,
                                            initial_state=initial_state)
        return doc_repr

    def score_net(self, state, action):
        # state => n_batch * HIDDEN_DIM
        # action => n_batch * n_candidate_doc * FEATURE_DIM
        # output:
        # score => n_batch * n_candidate_doc
        with tf.name_scope('score_net'):
            hidden_state = tf.layers.dense(tf.expand_dims(state, 1), FEATURE_DIM, reuse=tf.AUTO_REUSE,
                                           name='score_net_1')  # n_batch * 1 * FEATURE_DIM
            score = tf.squeeze(tf.matmul(hidden_state, tf.transpose(action, perm=[0, 2, 1]), name='score_net_2'), [1])
        return score

    def __init__(self):
        self.pos_input = tf.placeholder(tf.int32, name='pos_input')  # scalar
        self.query_input = tf.placeholder(tf.float32, [FEATURE_DIM], name='query_input')  # FEATURE_DIM
        self.data_input = tf.placeholder(tf.float32, [None, None, FEATURE_DIM],
                                         name='data_input')  # n_batch* n_doc * FEATURE_DIM
        self.reward_input = tf.placeholder(tf.float32, [None, None], name='reward_input')  # n_batch* n_sample
        self.action_id_input = tf.placeholder(tf.int32, [None, None], name='action_id_input')  # n_batch* n_sample

        self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_DIM, activation=tf.sigmoid, reuse=tf.AUTO_REUSE)

        self.ranked_docs = self.data_input[:, 0:self.pos_input, :]  # n_batch * n_ranked_docs * FEATURE_DIM
        self.candidate_docs = self.data_input[:, self.pos_input:, :]  # n_batch * n_candidate_docs * FEATURE_DIM

        n_batch = tf.shape(self.candidate_docs)[0]
        n_candidate_docs = tf.shape(self.candidate_docs)[1]

        # calculate the current_state
        query_repr = tf.tile(self.query_net(tf.expand_dims(self.query_input, 0)), [n_batch, 1])  # n_batch * hidden

        def get_query_state():
            return query_repr

        def get_doc_state():
            ranked_doc_repr = self.state_net(self.ranked_docs, query_repr)
            return ranked_doc_repr

        self.current_state = tf.cond(tf.equal(self.pos_input, 0), get_query_state, get_doc_state)  # batch*hidden

        # calculate the score
        self.score_all = self.score_net(self.current_state, self.candidate_docs)  # batch * n_candidate_docs

        # sample actions
        self.action_id_result = tf.transpose(tf.distributions.Categorical(probs=tf.nn.softmax(self.score_all)).sample(
            n_sample))  # n_batch * n_sample

        # simulate ranking
        n_iter = tf.constant(0, dtype=tf.int32)
        n_simulate = tf.cond(tf.less(top_n - self.pos_input - 1, n_candidate_docs - 1),
                             lambda: top_n - self.pos_input - 1,
                             lambda: n_candidate_docs - 1)

        def cond_fun_rank(iters, can_docs_rank, action_ids, state):
            return tf.less(iters, n_simulate)

        def loop_fun_rank(iters_rank, can_docs_rank, action_id_rank, state_rank):
            # can_docs_rank => (n_copy*batch) * n_candidate_doc_now * FEATURE_DIM
            # action_ids => (n_copy*batch)*1
            # state => (n_copy*batch) * HIDDEN_DIM
            n_candidate_doc_now = tf.shape(can_docs_rank)[1]
            # chosen docs
            score = self.score_net(state_rank, can_docs_rank)  # (n_copy*batch) * n_candidate_doc_now
            policy = tf.nn.softmax(score)
            dist_rank = tf.transpose(tf.distributions.Categorical(probs=policy).sample(1))  # (n_copy*batch) * 1
            idx = tf.stack([tf.range(n_copy * n_batch), tf.reshape(dist_rank, [-1])], axis=1)
            # update state
            chosen_doc_rank = tf.expand_dims(tf.gather_nd(can_docs_rank, idx), [1])  # (n_copy*batch) *1* FEATURE_DIM
            state_rank = self.state_net(chosen_doc_rank, state_rank)
            # update candidate docs
            mask = tf.equal(
                tf.scatter_nd(idx, tf.ones([n_copy * n_batch]), [n_copy * n_batch, n_candidate_doc_now]), 0)
            can_docs_rank = tf.reshape(tf.boolean_mask(can_docs_rank, mask),
                                       [n_copy * n_batch, -1, FEATURE_DIM])  # (n_copy*batch) * n_candidate_doc_now-1* FEATURE_DIM
            # update action id
            action_id_rank = tf.concat([action_id_rank, dist_rank], 1) # (n_copy*batch) * n_rank_docs
            iters_rank = tf.add(iters_rank, 1)
            return iters_rank, can_docs_rank, action_id_rank, state_rank

        def cond_fun(iters, action_ids):
            return tf.less(iters, n_sample)

        def loop_fun(iters, action_ids):
            doc_pos = self.action_id_result[:, iters]  # n_batch
            # update state
            chosen_doc = tf.expand_dims(tf.gather_nd(self.candidate_docs,
                                                     tf.stack([tf.range(n_batch), doc_pos], axis=1)),
                                        [1])  # n_batch *1* FEATURE_DIM
            current_state = self.state_net(chosen_doc, self.current_state)  # batch * hidden
            # update candidate docs
            mask = tf.not_equal(tf.tile(tf.expand_dims(tf.range(n_candidate_docs), 0), [n_batch, 1]),
                                tf.tile(tf.expand_dims(doc_pos, 1), [1, n_candidate_docs]))
            cdocs = tf.reshape(tf.boolean_mask(self.candidate_docs, mask),
                               [n_batch, -1, FEATURE_DIM])  # batch * n_candidate_doc-1 * FEATURE_DIM

            # copy data
            cdocs_copy = tf.reshape(tf.tile(tf.expand_dims(cdocs, 0), [n_copy, 1, 1, 1]),
                                    [n_copy * n_batch, -1, FEATURE_DIM])  # (n_copy*batch) * n_candidate_doc-1 * FEATURE_DIM
            doc_pos_copy = tf.reshape(tf.tile(tf.expand_dims(doc_pos, 0), [n_copy, 1]),
                                      [n_copy * n_batch, 1])  # (n_copy*batch)*1
            current_state_copy = tf.reshape(tf.tile(tf.expand_dims(current_state, 0), [n_copy, 1, 1]),
                                            [n_copy * n_batch, HIDDEN_DIM])  # (n_copy*batch) * HIDDEN_DIM

            _, _, action_ids_i, _ = tf.while_loop(cond_fun_rank, loop_fun_rank,
                                                  loop_vars=[n_iter, cdocs_copy, doc_pos_copy, current_state_copy],
                                                  shape_invariants=[n_iter.get_shape(),
                                                                    tf.TensorShape([None, None, FEATURE_DIM]),
                                                                    tf.TensorShape([None, None]),
                                                                    tf.TensorShape([None, HIDDEN_DIM])])

            action_ids = tf.cond(tf.equal(iters, 0), lambda: tf.expand_dims(action_ids_i, 0),
                                 lambda: tf.concat([action_ids, tf.expand_dims(action_ids_i, 0)], 0))
            iters = tf.add(iters, 1)

            return iters, action_ids

        len_list = tf.cond(tf.less(top_n - self.pos_input - 1, n_candidate_docs - 1),
                           lambda: top_n - self.pos_input - 1,
                           lambda: n_candidate_docs - 1)
        _, self.action_id_result = tf.while_loop(cond_fun, loop_fun, loop_vars=[n_iter,
                                                                                tf.zeros(
                                                                                    [1, n_copy * n_batch, len_list],
                                                                                    dtype=tf.int32)],
                                                 shape_invariants=[n_iter.get_shape(),
                                                                   tf.TensorShape([None, None, None])])

        self.loss = tf.reduce_mean(tf.multiply(tf.reshape(self.reward_input, [-1]),
                                               tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                   logits=tf.reshape(
                                                       tf.tile(tf.expand_dims(self.score_all, axis=1),
                                                               [1, n_sample, 1]),
                                                       [n_batch * n_sample, -1]),
                                                   labels=tf.reshape(self.action_id_input, [-1]))))

        # test

        candidate_docs_test = self.candidate_docs[0]

        def cond_fun_test(iters, cdocs, action_ids, state):
            len_list = tf.cond(tf.less(top_n - self.pos_input - 1, n_candidate_docs - 1),
                               lambda: top_n - self.pos_input - 1,
                               lambda: n_candidate_docs - 1)
            return tf.less(iters, len_list)

        def loop_fun_test(iters, cdocs_test, action_ids, state_test):
            # chose docs
            dist_test = tf.argmax(self.score_net(state_test, tf.expand_dims(cdocs_test, 0))[0], output_type=tf.int32)
            # update state
            state_test = self.state_net(tf.reshape(cdocs_test[dist_test], [-1, 1, FEATURE_DIM]), state_test)
            # update candidate docs
            cdocs_test = tf.concat([tf.slice(cdocs_test, [0, 0], [dist_test, FEATURE_DIM]),
                                    tf.slice(cdocs_test, [dist_test + 1, 0],
                                             [n_candidate_docs - dist_test - 2 - iters, FEATURE_DIM])], 0)
            action_ids = tf.concat([action_ids, tf.expand_dims(dist_test, 0)], 0)
            iters = tf.add(iters, 1)

            return iters, cdocs_test, action_ids, state_test

        dist_test = tf.argmax(self.score_net(query_repr, tf.expand_dims(candidate_docs_test, 0))[0],
                              output_type=tf.int32)
        state_test = self.state_net(tf.reshape(candidate_docs_test[dist_test], [-1, 1, FEATURE_DIM]), query_repr)
        cdocs = tf.concat([tf.slice(candidate_docs_test, [0, 0], [dist_test, FEATURE_DIM]),
                           tf.slice(candidate_docs_test, [dist_test + 1, 0],
                                    [n_candidate_docs - dist_test - 1, FEATURE_DIM])], 0)

        _, _, self.action_id_test_result, _ = tf.while_loop(cond_fun_test, loop_fun_test,
                                                            loop_vars=[0, cdocs, tf.expand_dims(dist_test, 0),
                                                                       state_test],
                                                            shape_invariants=[n_iter.get_shape(),
                                                                              cdocs.get_shape(),
                                                                              tf.TensorShape([None]),
                                                                              state_test.get_shape()])

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)

            # self.gradient = tf.train.AdamOptimizer(LR).compute_gradients(self.loss)
            # # tvars = tf.trainable_variables()
            # self.grads_set = [(tf.placeholder(dtype=tf.float32, shape=g.get_shape()), v) for (g, v) in self.gradient]
            # self.update = tf.train.AdamOptimizer(LR).apply_gradients(self.grads_set)

        gpuConfig = tf.ConfigProto(allow_soft_placement=True)
        gpuConfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpuConfig)
        self.sess.run(tf.global_variables_initializer())


    def interaction(self, query, data, pos):
        return self.sess.run(self.action_id_result,
                             {self.query_input: query, self.data_input: data, self.pos_input: pos})

    def interaction_test(self, query, data):
        # print 'test', np.shape(self.sess.run(self.test,
        #                                      {self.query_input: query, self.data_input: data, self.pos_input: 0}))
        return self.sess.run(self.action_id_test_result,
                             {self.query_input: query, self.data_input: data, self.pos_input: 0})

    def update_trajectories(self, trajectories):
        print '******************* begin update_trajectories ******************************'
        for data in trajectories.values():
            query = data['query_repr']
            for pos in data['info'].keys():
                action = data['info'][pos]['action']
                reward = data['info'][pos]['reward']
                action_id = data['info'][pos]['action_id']
                for idx in range(n_memory):
                    self.sess.run(self.train_op, {self.query_input: query,
                                                  self.data_input: [action[idx]],
                                                  self.pos_input: pos,
                                                  self.action_id_input: [action_id[idx]],
                                                  self.reward_input: [reward[idx]]})


class Reward(object):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def _dcg_reward_per_doc(self, query, doc, subtopic_count):
        if doc not in QUERY_DOC_TRUTH[query].keys():
            return 0
        gain = 0.0
        for subtopic_id in QUERY_DOC_TRUTH[query][doc]:
            gain += (1 - self.alpha) ** subtopic_count[subtopic_id]
            subtopic_count[subtopic_id] += 1
        return gain

    def reward_ranklist(self, query, doc_list, pos=0):
        pos_bias = pos + 2.0
        reward = []
        subtopic_count = Counter()
        for time, doc in enumerate(doc_list):
            doc_gain = self._dcg_reward_per_doc(query, doc, subtopic_count)
            reward.append(doc_gain / np.math.log(time + pos_bias, 2))
            time += 1
        return reward

    def sum_reward_ranklist(self, query, doc_list, pos=0):
        pos_bias = pos + 2.0
        sum_reward = 0.0
        subtopic_count = Counter()
        for time, doc in enumerate(doc_list):
            doc_gain = self._dcg_reward_per_doc(query, doc, subtopic_count)
            sum_reward += doc_gain / np.math.log(time + pos_bias, 2)
            time += 1
        return sum_reward

    def ndcg_ranklist(self, query, doc_ids, action_id_list, pos=0):
        n_doc = min(len(action_id_list), top_n)
        doc_id_list = self.get_doc_id_list(doc_ids[pos:], action_id_list)

        dcg_list = self.reward_ranklist(query, doc_id_list)
        for doc_pos in range(n_doc - 1):
            dcg_list[doc_pos + 1] += dcg_list[doc_pos]
        if n_doc < top_n:
            print 'small candidate'
            dcg_list += (top_n - n_doc) * [0]
        for doc_pos in range(top_n):
            if MAX_DCG[query][doc_pos] > 0:
                dcg_list[doc_pos] = dcg_list[doc_pos] / MAX_DCG[query][doc_pos]

        return dcg_list

    def get_doc_id_list(self, doc_ids, actoin_id_list):
        # n_copy * n_batch
        doc_id_list = []
        for action_id in actoin_id_list:
            doc_id_list.append(doc_ids[action_id])
            del doc_ids[action_id]
        return doc_id_list

    def get_return(self, query, doc_ids, action_id_list, pos):
        n_sample = np.shape(action_id_list)[0]
        n_copy = np.shape(action_id_list)[1]
        reward = []
        for i in range(n_sample):
            reward.append([])
            for j in range(n_copy):
                doc_id_list = self.get_doc_id_list(doc_ids[pos:], action_id_list[i, j, :])
                reward[i].append(self.sum_reward_ranklist(query, doc_id_list, pos))
        reward = np.asarray(reward)
        return np.mean(reward, 1)


class SearchEngine(object):
    def __init__(self):
        self.model = Ranker()
        self.reward_model = Reward()
        self.trajectories = {}
        self.data = {}

        self.init_data()
        self.init_trajectory()

    def init_data(self):
        print '***************   init data   ***************'
        for query in QUERY_DOC.keys():
            self.data[query] = {'query_repr': [], 'doc_repr': [], 'doc_id': []}
            self.data[query]['query_repr'] = QUERY_REPR[query]
            for doc in QUERY_DOC[query]:
                self.data[query]['doc_id'].append(doc)
                self.data[query]['doc_repr'].append(DOC_REPR[doc])

    def init_trajectory(self):
        print '***************   init trajectory   ***************'
        for query in QUERY_TRAIN:
            self.trajectories[query] = {'query_repr': self.data[query]['query_repr'], 'info': {}}
            n_doc = len(QUERY_DOC[query])
            len_list = min(n_doc, top_n)
            for pos in range(len_list):
                self.trajectories[query]['info'][pos] = {'action': [], 'reward': [], 'action_id': []}
            for _ in range(n_memory):
                query_repr = self.data[query]['query_repr']
                doc_repr = self.data[query]['doc_repr']
                doc_id = self.data[query]['doc_id']
                for pos in range(len_list):
                    action_id_result = self.model.interaction(query_repr, [doc_repr],
                                                              pos)  # n_sample * n_copy * len_list
                    self.trajectories[query]['info'][pos]['action'].append(doc_repr)
                    self.trajectories[query]['info'][pos]['action_id'].append(action_id_result[:, 0, 0])

                    reward = self.reward_model.get_return(query, doc_id, action_id_result, pos)
                    reward = reward - np.mean(reward)
                    self.trajectories[query]['info'][pos]['reward'].append(reward)

                    idx = action_id_result[np.argmax(reward), 0, 0]
                    tmp = doc_repr[idx]
                    doc_repr[idx] = doc_repr[pos]
                    doc_repr[pos] = tmp

                    tmp = doc_id[idx]
                    doc_id[idx] = doc_id[pos]
                    doc_id[pos] = tmp

    def gen_trajectory(self):
        print '******************* begin gen_trajectory ******************************'
        reward_sum = 0
        for query in QUERY_TRAIN:
            permutation = np.random.permutation(n_memory)
            len_list = min(len(QUERY_DOC[query]), top_n)

            for idx_pos in range(N_BATCH):
                idx = permutation[idx_pos]
                query_repr = self.data[query]['query_repr']
                doc_repr = self.data[query]['doc_repr']
                doc_id = self.data[query]['doc_id']

                for pos in range(len_list):
                    action_id_result = self.model.interaction(query_repr, [doc_repr], pos)
                    self.trajectories[query]['info'][pos]['action'][idx] = doc_repr
                    self.trajectories[query]['info'][pos]['action_id'][idx] = action_id_result[:, 0, 0]

                    reward = self.reward_model.get_return(query, doc_id, action_id_result, pos)
                    reward_sum += np.mean(reward)
                    reward = reward - np.mean(reward)
                    self.trajectories[query]['info'][pos]['reward'][idx] = reward

                    idx_max = action_id_result[np.argmax(reward), 0, 0]

                    tmp = doc_repr[idx_max]
                    doc_repr[idx_max] = doc_repr[pos]
                    doc_repr[pos] = tmp

                    tmp = doc_id[idx_max]
                    doc_id[idx_max] = doc_id[pos]
                    doc_id[pos] = tmp

        print 'reward_sum', reward_sum
        run_result = {'reward_sum': reward_sum}
        result_file.write(json.dumps(run_result) + '\n')

    # ***************************************  Test  *******************************************
    def test_model(self, query_list=QUERY_TEST, key='test'):
        measure_list = []
        for query in query_list:
            query_repr = self.data[query]['query_repr']
            doc_repr = self.data[query]['doc_repr']
            doc_id = self.data[query]['doc_id']
            action_id_result = self.model.interaction_test(query_repr, [doc_repr])
            measure_list.append(self.reward_model.ndcg_ranklist(query, doc_id, action_id_result))

        result = np.mean(np.asarray(measure_list), 0).tolist()
        run_result = {'key': key, 'ndcg': result}
        result_file.write(json.dumps(run_result) + '\n')
        print key, result[0], result[4], result[9]

    # ***************************************  Train  *******************************************#

    def train(self):
        self.test_model(QUERY_TEST, 'test')
        self.test_model(QUERY_TRAIN, 'train')

        for ite in range(20000):
            self.gen_trajectory()
            self.model.update_trajectories(self.trajectories)

            if ite % 1 == 0:
                print ite
                self.test_model(QUERY_TEST, 'test')
                self.test_model(QUERY_TRAIN, 'train')


SE_model = SearchEngine()
SE_model.train()
