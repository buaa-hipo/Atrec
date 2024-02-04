# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import time
import argparse
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import os
import sys
import math
import collections
from tensorflow.python.client import timeline
import json

from tensorflow.python.ops import partitioned_variables

from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.feature_column import utils as fc_utils

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
import blade_disc_tf as disc

global_time_cost = 0
global_auc = 0

tf.disable_v2_behavior()
disc.enable()
RNNCell = rnn_cell_impl.RNNCell
_WEIGHTS_VARIABLE_NAME = rnn_cell_impl._WEIGHTS_VARIABLE_NAME
_BIAS_VARIABLE_NAME = rnn_cell_impl._BIAS_VARIABLE_NAME

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants
UNSEQ_COLUMNS = ['UID', 'ITEM', 'CATEGORY']
HIS_COLUMNS = ['HISTORY_ITEM', 'HISTORY_CATEGORY']
NEG_COLUMNS = ['NOCLK_HISTORY_ITEM', 'NOCLK_HISTORY_CATEGORY']
SEQ_COLUMNS = HIS_COLUMNS + NEG_COLUMNS
LABEL_COLUMN = ['CLICKED']
TRAIN_DATA_COLUMNS = LABEL_COLUMN + UNSEQ_COLUMNS + SEQ_COLUMNS

EMBEDDING_DIM = 4
HIDDEN_SIZE = EMBEDDING_DIM * 2
ATTENTION_SIZE = EMBEDDING_DIM * 2
MAX_SEQ_LENGTH = 100

from tensorflow.python.platform import tf_logging as logging
logging.set_verbosity(logging.INFO)
class _Linear(object):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch, n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Raises:
    ValueError: if inputs_shape is wrong.
  """

  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
    self._build_bias = build_bias

    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape.dims[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape.dims[1].value

    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
      self._weights = vs.get_variable(
          _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if build_bias:
        with vs.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
          self._biases = vs.get_variable(
              _BIAS_VARIABLE_NAME, [output_size],
              dtype=dtype,
              initializer=bias_initializer)

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = math_ops.matmul(args[0], self._weights)
    else:
      # Explicitly creating a one for a minor performance improvement.
      one = constant_op.constant(1, dtype=dtypes.int32)
      res = math_ops.matmul(array_ops.concat(args, one), self._weights)
    if self._build_bias:
      res = nn_ops.bias_add(res, self._biases)
    return res

class VecAttGRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.math.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state):
        return self.call(inputs, state)

    def call(self, inputs, state, att_score=None):
        '''Gated recurrent unit (GRU) with nunits cells.'''
        _inputs = inputs[0]
        att_score = inputs[1]
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = tf.constant_initializer(1.0, dtype=_inputs.dtype)
            with tf.variable_scope('gates'):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [_inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = tf.math.sigmoid(self._gate_linear([_inputs, state]))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with tf.variable_scope('candidate'):
                self._candidate_linear = _Linear(
                    [_inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([_inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h


class DIEN():
    def __init__(self,
                 uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n,
                 learning_rate=0.001,
                 embedding_dim=18,
                 hidden_size=36,
                 attention_size=36,
                 inputs=None,
                 optimizer_type='adam',
                 bf16=False,
                 stock_tf=None,
                 emb_fusion=None,
                 ev=None,
                 ev_elimination=None,
                 ev_filter=None,
                 adaptive_emb=None,
                 dynamic_ev=None,
                 ev_opt=None,
                 multihash=None,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError('Dataset is not defined.')

        self.uid_n = uid_n
        self.item_n = item_n
        self.cate_n = cate_n
        self.shop_n = shop_n
        self.node_n = node_n
        self.product_n = product_n
        self.brand_n = brand_n

        self._feature = inputs[0]
        self._label = inputs[1]

        self.tf = stock_tf
        self.bf16 = False if self.tf else bf16
        self.is_training = True
        self._emb_fusion = emb_fusion
        self._adaptive_emb = adaptive_emb
        self._ev = ev
        self._ev_elimination = ev_elimination
        self._ev_filter = ev_filter
        self._dynamic_ev = dynamic_ev
        self._ev_opt = ev_opt
        self._multihash = multihash

        self._learning_rate = learning_rate
        self._optimizer_type = optimizer_type
        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner

        self._batch_size = tf.shape(self._label)[0]
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._attention_size = attention_size
        self._data_type = tf.bfloat16 if self.bf16 else tf.float32

        self._create_model()
        with tf.name_scope('head'):
            self._create_loss()
            self._create_optimizer()
            self._create_metrics()

    def _embedding_input_layer(self):
        def get_embeddings_variable(var_name, embedding_shape):
            # workaround to return vector of 0
            embeddings = tf.get_variable(var_name, embedding_shape, trainable=True)
            embeddings = tf.concat([ embeddings, [tf.constant([0.] * embedding_shape[1])] ], axis = 0)
            return embeddings 

        with tf.name_scope('Inputs'):

            # for k in filter(lambda x: 'history' in x, self._feature.keys()):
            for k in self._feature.keys():
                self._feature[k] = tf.where(tf.greater(self._feature[k], 0), self._feature[k], tf.zeros_like(self._feature[k]))

            self._item_id_his_batch = self._feature['history_item_array']
            self._cate_his_batch = self._feature['history_cate_array']
            self._shop_his_batch = self._feature['history_shop_array']
            self._node_his_batch = self._feature['history_node_array']
            self._product_his_batch = self._feature['history_product_array']
            self._brand_his_batch = self._feature['history_brand_array']
            self._uid_batch = self._feature['uid']
            self._item_id_batch = self._feature['item']
            self._cate_id_batch = self._feature['cate']
            self._shop_id_batch = self._feature['shop']
            self._node_id_batch = self._feature['node']
            self._product_id_batch = self._feature['product']
            self._brand_id_batch = self._feature['brand']
            self._mask = self._feature['history_mask']

            self._item_id_neg_batch = self._feature['neg_history_item_array']
            self._cate_neg_batch = self._feature['neg_history_cate_array']
            self._shop_neg_batch = self._feature['neg_history_shop_array']
            self._node_neg_batch = self._feature['neg_history_node_array']
            self._product_neg_batch = self._feature['neg_history_product_array']
            self._brand_neg_batch = self._feature['neg_history_brand_array']

            self._sequence_length = tf.shape(self._item_id_his_batch)[1]

        # Embedding layer
        with tf.name_scope('Embedding_layer'):

            #self.item_id_embeddings_var = tf.get_variable("item_id_embedding_var", [item_n, EMBEDDING_DIM], trainable=True)
            self._item_id_embeddings_var = get_embeddings_variable("item_id_embedding_var", [self.item_n, EMBEDDING_DIM])
            self._item_id_batch_embedded = tf.nn.embedding_lookup(self._item_id_embeddings_var, self._item_id_batch)
            self._item_id_his_batch_embedded = tf.nn.embedding_lookup(self._item_id_embeddings_var, self._item_id_his_batch)
           
            self._cate_id_embeddings_var = tf.get_variable("cate_id_embedding_var", [self.cate_n, EMBEDDING_DIM], trainable=True)
            self._cate_id_batch_embedded = tf.nn.embedding_lookup(self._cate_id_embeddings_var, self._cate_id_batch)
            self._cate_his_batch_embedded = tf.nn.embedding_lookup(self._cate_id_embeddings_var, self._cate_his_batch)

            #self.shop_id_embeddings_var = tf.get_variable("shop_id_embedding_var", [shop_n, EMBEDDING_DIM], trainable=True)
            self._shop_id_embeddings_var = get_embeddings_variable("shop_id_embedding_var", [self.shop_n, EMBEDDING_DIM])
            self._shop_id_batch_embedded = tf.nn.embedding_lookup(self._shop_id_embeddings_var, self._shop_id_batch)
            self._shop_his_batch_embedded = tf.nn.embedding_lookup(self._shop_id_embeddings_var, self._shop_his_batch)

            self._node_id_embeddings_var = tf.get_variable("node_id_embedding_var", [self.node_n, EMBEDDING_DIM], trainable=True)
            self._node_id_batch_embedded = tf.nn.embedding_lookup(self._node_id_embeddings_var, self._node_id_batch)
            self._node_his_batch_embedded = tf.nn.embedding_lookup(self._node_id_embeddings_var, self._node_his_batch)
            
            self._product_id_embeddings_var = tf.get_variable("product_id_embedding_var", [self.product_n, EMBEDDING_DIM], trainable=True)
            self._product_id_batch_embedded = tf.nn.embedding_lookup(self._product_id_embeddings_var, self._product_id_batch)
            self._product_his_batch_embedded = tf.nn.embedding_lookup(self._product_id_embeddings_var, self._product_his_batch)
            
            #self.brand_id_embeddings_var = tf.get_variable("brand_id_embedding_var", [self.brand_n, EMBEDDING_DIM], trainable=True)
            self._brand_id_embeddings_var = get_embeddings_variable("brand_id_embedding_var", [self.brand_n, EMBEDDING_DIM])
            self._brand_id_batch_embedded = tf.nn.embedding_lookup(self._brand_id_embeddings_var, self._brand_id_batch)
            self._brand_his_batch_embedded = tf.nn.embedding_lookup(self._brand_id_embeddings_var, self._brand_his_batch)  

            self._neg_item_his_eb = tf.nn.embedding_lookup(self._item_id_embeddings_var, self._item_id_neg_batch)
            self._neg_cate_his_eb = tf.nn.embedding_lookup(self._cate_id_embeddings_var, self._cate_neg_batch)
            self._neg_shop_his_eb = tf.nn.embedding_lookup(self._shop_id_embeddings_var, self._shop_neg_batch)
            self._neg_node_his_eb = tf.nn.embedding_lookup(self._node_id_embeddings_var, self._node_neg_batch)
            self._neg_product_his_eb = tf.nn.embedding_lookup(self._product_id_embeddings_var, self._product_neg_batch)
            self._neg_brand_his_eb = tf.nn.embedding_lookup(self._brand_id_embeddings_var, self._brand_neg_batch)

            neg_his_eb = tf.concat([self._neg_item_his_eb,self._neg_cate_his_eb, self._neg_shop_his_eb, self._neg_node_his_eb, self._neg_product_his_eb, self._neg_brand_his_eb], axis=2) * tf.reshape(self._mask,(self._batch_size, self._sequence_length, 1))   

            item_eb = tf.concat([self._item_id_batch_embedded, self._cate_id_batch_embedded, self._shop_id_batch_embedded, self._node_id_batch_embedded, self._product_id_batch_embedded, self._brand_id_batch_embedded], axis=1)
            item_his_eb = tf.concat([self._item_id_his_batch_embedded,self._cate_his_batch_embedded, self._shop_his_batch_embedded, self._node_his_batch_embedded, self._product_his_batch_embedded, self._brand_his_batch_embedded], axis=2) * tf.reshape(self._mask,(self._batch_size, self._sequence_length, 1))
            #debug if last item of history is leaked
            #self.item_his_eb = self.item_his_eb[:,:-1,:]

        return None, item_eb, item_his_eb, neg_his_eb

    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    def _assert_all_equal_and_return(self, tensors, name=None):
        '''Asserts that all tensors are equal and returns the first one.'''
        with tf.name_scope(name, 'assert_all_equal', values=tensors):
            if len(tensors) == 1:
                return tensors[0]
            assert_equal_ops = []
            for t in tensors[1:]:
                assert_equal_ops.append(
                    tf.debugging.assert_equal(tensors[0], t))
            with tf.control_dependencies(assert_equal_ops):
                return tf.identity(tensors[0])

    def _prelu(self, x, scope=''):
        '''parametric ReLU activation'''
        with tf.variable_scope(name_or_scope=scope, default_name='prelu'):
            alpha = tf.get_variable('prelu_' + scope,
                                    shape=x.get_shape()[-1],
                                    dtype=x.dtype,
                                    initializer=tf.constant_initializer(0.1))
            pos = tf.nn.relu(x)
            neg = alpha * (x - abs(x)) * tf.constant(0.5, dtype=x.dtype)
            return pos + neg

    def _auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_,
                                            name='bn1' + stag,
                                            reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1,
                               100,
                               activation=None,
                               name='f1' + stag,
                               reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1,
                               50,
                               activation=None,
                               name='f2' + stag,
                               reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2,
                               2,
                               activation=None,
                               name='f3' + stag,
                               reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + tf.constant(0.00000001, dtype=dnn3.dtype)
        return y_hat

    def _auxiliary_loss(self,
                        h_states,
                        click_seq,
                        noclick_seq,
                        mask,
                        dtype=tf.float32,
                        stag=None):
        mask = tf.cast(mask, dtype=dtype)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        with tf.variable_scope('auxiliary_net'):
            click_prop_ = self._auxiliary_net(click_input_, stag=stag)[:, :, 0]
            noclick_prop_ = self._auxiliary_net(noclick_input_,
                                                stag=stag)[:, :, 0]

        click_loss_ = -tf.reshape(tf.log(click_prop_),
                                  [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = -tf.reshape(tf.log(1.0 - noclick_prop_),
                                    [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def _attention(self,
                   query,
                   facts,
                   attention_size,
                   mask,
                   stag='null',
                   mode='SUM',
                   softmax_stag=1,
                   time_major=False,
                   return_alphas=False,
                   forCnn=False):
        if isinstance(facts, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            facts = tf.concat(facts, 2)
        if len(facts.get_shape().as_list()) == 2:
            facts = tf.expand_dims(facts, 1)

        if time_major:  # (T,B,D) => (B,T,D)
            facts = tf.array_ops.transpose(facts, [1, 0, 2])
        # Trainable parameters
        mask = tf.equal(mask, tf.ones_like(mask))
        facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
        query_size = query.get_shape().as_list()[-1]
        query = tf.layers.dense(query,
                                facts_size,
                                activation=None,
                                name='f1' + stag)
        query = self._prelu(query)
        queries = tf.tile(query, [1, tf.shape(facts)[1]])
        queries = tf.reshape(queries, tf.shape(facts))
        din_all = tf.concat([queries, facts, queries - facts, queries * facts],
                            axis=-1)
        d_layer_1_all = tf.layers.dense(din_all,
                                        80,
                                        activation=tf.nn.sigmoid,
                                        name='f1_att' + stag)
        d_layer_2_all = tf.layers.dense(d_layer_1_all,
                                        40,
                                        activation=tf.nn.sigmoid,
                                        name='f2_att' + stag)
        d_layer_3_all = tf.layers.dense(d_layer_2_all,
                                        1,
                                        activation=None,
                                        name='f3_att' + stag)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
        scores = d_layer_3_all
        # Mask
        # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
        key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2**32 + 1)
        if not forCnn:
            scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

        # Scale
        # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

        # Activation
        if softmax_stag:
            scores = tf.nn.softmax(scores)  # [B, 1, T]

        # Weighted sum
        if mode == 'SUM':
            output = tf.matmul(scores, facts)  # [B, 1, H]
            # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
        else:
            scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
            output = facts * tf.expand_dims(scores, -1)
            output = tf.reshape(output, tf.shape(facts))
        if return_alphas:
            return output, scores
        return output

    def _top_fc_layer(self, inputs):
        bn1 = tf.layers.batch_normalization(inputs=inputs, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='dnn1')
        dnn1 = tf.nn.relu(dnn1)

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='dnn2')
        dnn2 = tf.nn.relu(dnn2)

        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='dnn3')
        logits = tf.layers.dense(dnn3, 1, activation=None, name='logits')
        self._add_layer_summary(dnn1, 'dnn1')
        self._add_layer_summary(dnn2, 'dnn2')
        self._add_layer_summary(dnn3, 'dnn3')
        return logits


    # create model
    def _create_model(self):
        # input layer to get embedding of features
        with tf.variable_scope('input_layer',
                               partitioner=self._input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            _, item_eb, item_his_eb, neg_his_eb = self._embedding_input_layer()

            item_his_eb_sum = tf.reduce_sum(item_his_eb, 1)


        sequence_length = tf.fill([self._batch_size], self._sequence_length)

        # RNN layer_1
        with tf.variable_scope('rnn_1'):
            run_output_1, _ = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(self._hidden_size),
                inputs=item_his_eb,
                sequence_length=sequence_length,
                dtype=self._data_type,
                scope='gru1')
            tf.summary.histogram('GRU_outputs', run_output_1)

        # Aux loss
        aux_loss_scope = tf.variable_scope(
            'aux_loss', partitioner=self._dense_layer_partitioner)
        with aux_loss_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else aux_loss_scope:
            self._aux_loss = self._auxiliary_loss(run_output_1[:, :-1, :],
                                                  item_his_eb[:, 1:, :],
                                                  neg_his_eb[:, 1:, :],
                                                  self._mask[:, 1:],
                                                  dtype=self._data_type,
                                                  stag='gru')

        # Attention layer
        attention_scope = tf.variable_scope('attention_layer')
        with attention_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else attention_scope:
            _, alphas = self._attention(item_eb,
                                        run_output_1,
                                        self._attention_size,
                                        self._mask,
                                        softmax_stag=1,
                                        stag='1_1',
                                        mode='LIST',
                                        return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        # RNN layer_2
        run_output_1 = tf.reshape(run_output_1, [-1, tf.shape(run_output_1)[1], run_output_1.shape[2]]) # 太傻逼了，为什么静态维和动态维的组合不能过rnn的长度检测？反而让我故意把静态维搞成动态维？

        with tf.variable_scope('rnn_2'):
            _, final_state2 = tf.nn.dynamic_rnn(
                VecAttGRUCell(self._hidden_size),
                inputs=[run_output_1, tf.expand_dims(alphas, -1)],
                sequence_length=sequence_length,
                dtype=self._data_type,
                scope='gru2')
            tf.summary.histogram('GRU2_Final_State', final_state2)

        top_input = tf.concat([
            item_eb, item_his_eb_sum, item_eb * item_his_eb_sum,
            final_state2
        ], 1)

        # Top MLP layer
        top_mlp_scope = tf.variable_scope(
            'top_mlp_layer',
            partitioner=self._dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with top_mlp_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else top_mlp_scope:
            self._logits = self._top_fc_layer(top_input)
        if self.bf16:
            self._logits = tf.cast(self._logits, dtype=tf.float32)

        self.probability = tf.math.sigmoid(self._logits)
        self.output = tf.round(self.probability)

    # compute loss
    def _create_loss(self):
        self._logits = tf.squeeze(self._logits)
        self._crt_loss = tf.losses.sigmoid_cross_entropy(
            self._label,
            self._logits,
            scope='loss',
            reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.loss = self._crt_loss + self._aux_loss
        tf.summary.scalar('sigmoid_cross_entropy', self._crt_loss)
        tf.summary.scalar('aux_loss', self._aux_loss)
        tf.summary.scalar('loss', self.loss)

    # define optimizer and generate train_op
    def _create_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        if self.tf or self._optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._learning_rate)
        elif self._optimizer_type == 'adamasync':
            optimizer = tf.train.AdamAsyncOptimizer(
                learning_rate=self._learning_rate)
        elif self._optimizer_type == 'adagraddecay':
            optimizer = tf.train.AdagradDecayOptimizer(
                learning_rate=self._learning_rate,
                global_step=self.global_step)
        else:
            raise ValueError("Optimzier type error.")

        gradients = optimizer.compute_gradients(self.loss)
        clipped_gradients = [(tf.clip_by_norm(grad, 5), var)
                             for grad, var in gradients if grad is not None]

        self.train_op = optimizer.apply_gradients(clipped_gradients,
                                                  global_step=self.global_step)

    # compute acc & auc
    def _create_metrics(self):
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self._label,
                                                    predictions=self.output)
        self.auc, self.auc_op = tf.metrics.auc(labels=self._label,
                                               predictions=self.probability,
                                               num_thresholds=1000)
        tf.summary.scalar('eval_acc', self.acc)
        tf.summary.scalar('eval_auc', self.auc)


# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs):

    def parse_npz(values):
        tf.logging.info('Parsing {}'.format(filename))

        feature_map = {
            'history_item_array': tf.FixedLenFeature((100), tf.int64),
            'neg_history_item_array': tf.FixedLenFeature((100), tf.int64),
            'history_item_array': tf.FixedLenFeature((100), tf.int64),
            'neg_history_item_array': tf.FixedLenFeature((100), tf.int64),
            'history_cate_array': tf.FixedLenFeature((100), tf.int64),
            'neg_history_cate_array': tf.FixedLenFeature((100), tf.int64),
            'history_shop_array': tf.FixedLenFeature((100), tf.int64),
            'neg_history_shop_array': tf.FixedLenFeature((100), tf.int64),
            'history_node_array': tf.FixedLenFeature((100), tf.int64),
            'neg_history_node_array': tf.FixedLenFeature((100), tf.int64),
            'history_product_array': tf.FixedLenFeature((100), tf.int64),
            'neg_history_product_array': tf.FixedLenFeature((100), tf.int64),
            'history_brand_array': tf.FixedLenFeature((100), tf.int64),
            'neg_history_brand_array': tf.FixedLenFeature((100), tf.int64),
            'target_array': tf.FixedLenFeature((2), tf.int64),
            'source_array': tf.FixedLenFeature((7), tf.int64),
        }
        all_columns = tf.parse_example(values, features=feature_map)
        cond = tf.math.greater(all_columns['history_item_array'], 0)
        all_columns['history_mask'] = tf.cast(cond, tf.float32)
        source_array = all_columns.pop('source_array')
        uid, item, cate, shop, node, product, brand = tf.split(source_array, [1, 1, 1, 1, 1, 1, 1], axis=1)
        all_columns['uid'] = tf.reshape(uid, [-1])
        all_columns['item'] = tf.reshape(item, [-1])
        all_columns['cate'] = tf.reshape(cate, [-1])
        all_columns['shop'] = tf.reshape(shop, [-1])
        all_columns['node'] = tf.reshape(node, [-1])
        all_columns['product'] = tf.reshape(product, [-1])
        all_columns['brand'] = tf.reshape(brand, [-1])

        labels = all_columns.pop('target_array')[:,0]
        features = all_columns

        return features, labels

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.shuffle(buffer_size=20000, seed=args.seed)  # set seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_npz, num_parallel_calls=1)
    dataset = dataset.prefetch(2)
    return dataset


def train(sess_config,
          input_hooks,
          model,
          data_init_op,
          steps,
          checkpoint_dir,
          tf_config=None,
          server=None):
    model.is_training = True
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.tables_initializer(),
                               tf.local_variables_initializer(), data_init_op),
        saver=tf.train.Saver(max_to_keep=args.keep_checkpoint_max))

    stop_hook = tf.train.StopAtStepHook(last_step=steps)
    log_hook = tf.train.LoggingTensorHook(
        {
            'steps': model.global_step,
            'loss': model.loss
        }, every_n_iter=100)
    hooks.append(stop_hook)
    hooks.append(log_hook)
    if args.timeline > 0:
        hooks.append(
            tf.train.ProfilerHook(save_steps=args.timeline,
                                  output_dir=checkpoint_dir))
    save_steps = args.save_steps if args.save_steps or args.no_eval else steps
    '''
                            Incremental_Checkpoint
    Please add `save_incremental_checkpoint_secs` in 'tf.train.MonitoredTrainingSession'
    it's default to None, Incremental_save checkpoint time in seconds can be set 
    to use incremental checkpoint function, like `tf.train.MonitoredTrainingSession(
        save_incremental_checkpoint_secs=args.incremental_ckpt)`
    '''
    if args.incremental_ckpt and not args.tf:
        print("Incremental_Checkpoint is not really enabled.")
        print("Please see the comments in the code.")
        sys.exit()

    time_start = time.perf_counter()
    with tf.train.MonitoredTrainingSession(
            master=server.target if server else '',
            is_chief=tf_config['is_chief'] if tf_config else True,
            hooks=hooks,
            scaffold=scaffold,
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_steps=steps,
            summary_dir=checkpoint_dir,
            save_summaries_steps=None,
            config=sess_config) as sess:
        while not sess.should_stop():
            sess.run([model.loss, model.train_op])
    time_end = time.perf_counter()
    print("Training completed.")
    time_cost = time_end - time_start
    global global_time_cost
    global_time_cost = time_cost


def eval(sess_config, input_hooks, model, data_init_op, steps, checkpoint_dir):
    model.is_training = False
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.tables_initializer(),
                               tf.local_variables_initializer(), data_init_op))
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold, checkpoint_dir=checkpoint_dir, config=sess_config)
    writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'eval'))
    merged = tf.summary.merge_all()

    with tf.train.MonitoredSession(session_creator=session_creator,
                                   hooks=hooks) as sess:
        for _in in range(1, steps + 1):
            if (_in != steps):
                sess.run([model.acc_op, model.auc_op])
                if (_in % 100 == 0):
                    print("Evaluation complate:[{}/{}]".format(_in, steps))
            else:
                eval_acc, eval_auc, events = sess.run(
                    [model.acc_op, model.auc_op, merged])
                writer.add_summary(events, _in)
                print("Evaluation complate:[{}/{}]".format(_in, steps))
                print("ACC = {}\nAUC = {}".format(eval_acc, eval_auc))
                global global_auc
                global_auc = eval_auc


def main(tf_config=None, server=None):
    # check dataset and count data set size
    disc.enable()
    print("Checking dataset...")
    train_file = args.data_location + '/train_sample_0.tfrecords'
    test_file = args.data_location + '/test_sample_0.tfrecords'
    
    no_of_training_examples = sum(1 for _ in tf.python_io.tf_record_iterator(train_file))
    no_of_test_examples = sum(1 for _ in tf.python_io.tf_record_iterator(test_file))
    print("Numbers of training dataset is {}".format(no_of_training_examples))
    print("Numbers of test dataset is {}".format(no_of_test_examples))

    # set batch size, eporch & steps
    batch_size = math.ceil(
        args.batch_size / args.micro_batch
    ) if args.micro_batch and not args.tf else args.batch_size

    if args.steps == 0:
        no_of_epochs = 1
        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size)
    else:
        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)
    print("The training steps is {}".format(train_steps))
    print("The testing steps is {}".format(test_steps))

    # set fixed random seed
    tf.set_random_seed(args.seed)

    # set directory path for checkpoint_dir
    model_dir = os.path.join(args.output_dir,
                             'model_DIEN_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)

    # create data pipline of train & test dataset
    train_dataset = build_model_input(train_file, batch_size, no_of_epochs)
    test_dataset = build_model_input(test_file, batch_size, 1)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               test_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # create variable partitioner for distributed training
    num_ps_replicas = len(tf_config['ps_hosts']) if tf_config else 0
    input_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=args.input_layer_partitioner <<
        20) if args.input_layer_partitioner else None
    dense_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=args.dense_layer_partitioner <<
        10) if args.dense_layer_partitioner else None

    # Session config
    sess_config = tf.ConfigProto()
    sess_config.inter_op_parallelism_threads = args.inter
    sess_config.intra_op_parallelism_threads = args.intra

    # Session hooks
    hooks = []

    if args.smartstaged and not args.tf:
        '''Smart staged Feature'''
        next_element = tf.staged(next_element, num_threads=4, capacity=40)
        sess_config.graph_options.optimizer_options.do_smart_stage = True
        hooks.append(tf.make_prefetch_hook())
    if args.op_fusion and not args.tf:
        '''Auto Graph Fusion'''
        sess_config.graph_options.optimizer_options.do_op_fusion = True
    if args.micro_batch and not args.tf:
        '''Auto Mirco Batch'''
        sess_config.graph_options.optimizer_options.micro_batch_num = args.micro_batch

    uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n = [7956430, 34196611, 5596, 4377722, 2975349, 65624, 584181] 
    # item_n = 9759820
    # cate_n = 5129
    # shop_n = 1785060
    # node_n = 1377624
    # product_n = 46003
    # brand_n = 390116
    # item_n = 10000
    # cate_n = 5129
    # shop_n = 10000
    # node_n = 10000
    # product_n = 10000
    # brand_n = 10000

    # create model
    model = DIEN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n,
                 learning_rate=args.learning_rate,
                 embedding_dim=EMBEDDING_DIM,
                 hidden_size=HIDDEN_SIZE,
                 attention_size=ATTENTION_SIZE,
                 optimizer_type=args.optimizer,
                 bf16=args.bf16,
                 stock_tf=args.tf,
                 emb_fusion=args.emb_fusion,
                 ev=args.ev,
                 ev_elimination=args.ev_elimination,
                 ev_filter=args.ev_filter,
                 adaptive_emb=args.adaptive_emb,
                 dynamic_ev=args.dynamic_ev,
                 inputs=next_element,
                 multihash=args.multihash,
                 input_layer_partitioner=input_layer_partitioner,
                 dense_layer_partitioner=dense_layer_partitioner)

    # Run model training and evaluation
    train(sess_config, hooks, model, train_init_op, train_steps,
          checkpoint_dir, tf_config, server)
    if not (args.no_eval or tf_config):
        eval(sess_config, hooks, model, test_init_op, test_steps,
             checkpoint_dir)
    print(global_auc)
    print(global_time_cost)


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return low_string == 'true'


# Get parse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=2700)
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to model output directory. \
                            Default to ./result. Covered by --checkpoint. ',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output. \
                            Default to ./result/$MODEL_TIMESTAMP',
                        required=False)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--seed',
                        help='set the random seed for tensorflow',
                        type=int,
                        default=2021)
    parser.add_argument('--optimizer',
                        type=str, \
                        choices=['adam', 'adamasync', 'adagraddecay'],
                        default='adamasync')
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.001)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline. Default 0',
                        type=int,
                        default=0)
    parser.add_argument('--protocol',
                        type=str,
                        choices=['grpc', 'grpc++', 'star_server'],
                        default='grpc')
    parser.add_argument('--inter',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--input_layer_partitioner', \
                        help='slice size of input layer partitioner, units MB. Default 8MB',
                        type=int,
                        default=8)
    parser.add_argument('--dense_layer_partitioner', \
                        help='slice size of dense layer partitioner, units KB. Default 16KB',
                        type=int,
                        default=16)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    parser.add_argument('--tf', \
                        help='Use TF 1.15.5 API and disable DeepRec feature to run a baseline.',
                        action='store_true')
    parser.add_argument('--smartstaged', \
                        help='Whether to enable smart staged feature of DeepRec, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--emb_fusion', \
                        help='Whether to enable embedding fusion, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--ev', \
                        help='Whether to enable DeepRec EmbeddingVariable. Default False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--ev_elimination', \
                        help='Feature Elimination of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'l2', 'gstep'],
                        default=None)
    parser.add_argument('--ev_filter', \
                        help='Feature Filter of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'counter', 'cbf'],
                        default=None)
    parser.add_argument('--op_fusion', \
                        help='Whether to enable Auto graph fusion feature. Default to True',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--micro_batch',
                        help='Set num for Auto Mirco Batch. Default close.',
                        type=int,
                        default=0)  #TODO: Defautl to True
    parser.add_argument('--adaptive_emb', \
                        help='Whether to enable Adaptive Embedding. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--dynamic_ev', \
                        help='Whether to enable Dynamic-dimension Embedding Variable. Default to False.',
                        type=boolean_string,
                        default=False)#TODO
    parser.add_argument('--incremental_ckpt', \
                        help='Set time of save Incremental Checkpoint. Default 0 to close.',
                        type=int,
                        default=0)
    parser.add_argument('--workqueue', \
                        help='Whether to enable Work Queue. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--multihash', \
                        help='Whether to enable Multi-Hash Variable. Default to False.',
                        type=boolean_string,
                        default=False)#TODO
    return parser


# Parse distributed training configuration and generate cluster information
def generate_cluster_info(TF_CONFIG):
    print(TF_CONFIG)
    tf_config = json.loads(TF_CONFIG)
    cluster_config = tf_config.get('cluster')
    ps_hosts = []
    worker_hosts = []
    chief_hosts = []
    for key, value in cluster_config.items():
        if 'ps' == key:
            ps_hosts = value
        elif 'worker' == key:
            worker_hosts = value
        elif 'chief' == key:
            chief_hosts = value
    if chief_hosts:
        worker_hosts = chief_hosts + worker_hosts

    if not ps_hosts or not worker_hosts:
        print('TF_CONFIG ERROR')
        sys.exit()
    task_config = tf_config.get('task')
    task_type = task_config.get('type')
    task_index = task_config.get('index') + (1 if task_type == 'worker'
                                             and chief_hosts else 0)

    if task_type == 'chief':
        task_type = 'worker'

    is_chief = True if task_index == 0 else False
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.distribute.Server(cluster,
                                  job_name=task_type,
                                  task_index=task_index,
                                  protocol=args.protocol)
    if task_type == 'ps':
        server.join()
    elif task_type == 'worker':
        tf_config = {
            'ps_hosts': ps_hosts,
            'worker_hosts': worker_hosts,
            'type': task_type,
            'index': task_index,
            'is_chief': is_chief
        }
        tf_device = tf.device(
            tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % task_index,
                cluster=cluster))
        return tf_config, server, tf_device
    else:
        print("Task type or index error.")
        sys.exit()


# Some DeepRec's features are enabled by ENV.
# This func is used to set ENV and enable these features.
# A triple quotes comment is used to introduce these features and play an emphasizing role.
def set_env_for_DeepRec():
    '''
    Set some ENV for these DeepRec's features enabled by ENV. 
    More Detail information is shown in https://deeprec.readthedocs.io/zh/latest/index.html.
    START_STATISTIC_STEP & STOP_STATISTIC_STEP: On CPU platform, DeepRec supports memory optimization
        in both stand-alone and distributed trainging. It's default to open, and the 
        default start and stop steps of collection is 1000 and 1100. Reduce the initial 
        cold start time by the following settings.
    MALLOC_CONF: On CPU platform, DeepRec can use memory optimization with the jemalloc library.
        Please preload libjemalloc.so by `LD_PRELOAD=./libjemalloc.so.2 python ...`
    '''
    os.environ['START_STATISTIC_STEP'] = '100'
    os.environ['STOP_STATISTIC_STEP'] = '110'
    os.environ['MALLOC_CONF']= \
        'background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000'
    os.environ['ENABLE_MEMORY_OPTIMIZATION'] = '0'


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    if not args.tf:
        set_env_for_DeepRec()

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        main()
    else:
        tf_config, server, tf_device = generate_cluster_info(TF_CONFIG)
        main(tf_config, server)

