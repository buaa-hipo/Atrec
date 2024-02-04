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
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import numpy as np

import time
import argparse
import keras
import os
import sys
import math
import collections
import logging
import tensorflow as tf
from collections.abc import Mapping, Sequence
from deepctr_torch.models import dien, wdl, deepfm
from itertools import islice

logger = logging.getLogger(__name__)

from tensorflow.python.feature_column import utils as fc_utils

result_dir = "/tmp/tianchi/result/DIEN/"
result_path = result_dir + "result"
global_time_cost = 0
global_auc = 0

# Definition of some constants
UNSEQ_COLUMNS = ["UID", "ITEM", "CATEGORY"]
HIS_COLUMNS = ["HISTORY_ITEM", "HISTORY_CATEGORY"]
NEG_COLUMNS = ["NOCLK_HISTORY_ITEM", "NOCLK_HISTORY_CATEGORY"]
SEQ_COLUMNS = HIS_COLUMNS + NEG_COLUMNS
LABEL_COLUMN = ["CLICKED"]
TRAIN_DATA_COLUMNS = LABEL_COLUMN + UNSEQ_COLUMNS + SEQ_COLUMNS

USER_COLUMNS = ["UID"] + HIS_COLUMNS + NEG_COLUMNS
ITEM_COLUMNS = ["ITEM", "CATEGORY"]

EMBEDDING_DIM = 4
ITEM_ALL_EMBEDDING_DIM = EMBEDDING_DIM * len(ITEM_COLUMNS)
HIDDEN_SIZE = EMBEDDING_DIM * 2
ATTENTION_SIZE = EMBEDDING_DIM * 2
MAX_SEQ_LENGTH = 50
ITEM_PACKSIZE = 5000


# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs):
    def parse_csv(value, neg_value):
        logger.info("Parsing {}".format(filename))
        cate_defaults = [[" "] for i in range(0, 5)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS
        record_defaults = label_defaults + cate_defaults
        columns = tf.io.decode_csv(
            value, record_defaults=record_defaults, field_delim="\t"
        )
        neg_columns = tf.io.decode_csv(
            neg_value, record_defaults=[[""], [""]], field_delim="\t"
        )
        columns.extend(neg_columns)
        all_columns = collections.OrderedDict(zip(column_headers, columns))

        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        return features, labels

    files = filename
    neg_files = filename + "_neg"
    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(files)
    dataset_neg_samples = tf.data.TextLineDataset(neg_files)
    dataset = tf.data.Dataset.zip((dataset, dataset_neg_samples))
    dataset = dataset.shuffle(
        buffer_size=20000, seed=args.seed
    )  # set seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(2)
    return dataset


class FeaturePreprocessor:
    def __init__(self, data_location):
        self._data_location = data_location
        self._uid_file = os.path.join(data_location, "uid_voc.txt")
        self._mid_file = os.path.join(data_location, "mid_voc.txt")
        self._cat_file = os.path.join(data_location, "cat_voc.txt")

        if (
            (not os.path.exists(self._uid_file))
            or (not os.path.exists(self._mid_file))
            or (not os.path.exists(self._cat_file))
        ):
            print("uid_voc.txt, mid_voc.txt or cat_voc does not exist in data file.")
            sys.exit()

        self._uid_lookup = keras.layers.StringLookup(
            vocabulary=self._uid_file,
            output_mode="int",
        )

        self._item_lookup = keras.layers.StringLookup(
            vocabulary=self._mid_file,
            output_mode="int",
        )
        self._category_lookup = keras.layers.StringLookup(
            vocabulary=self._cat_file,
            output_mode="int",
        )

    def forward(self, features):
        for key in SEQ_COLUMNS:
            features[key] = tf.strings.split(features[key], "")[:, :MAX_SEQ_LENGTH]

        features["UID"] = self._uid_lookup(features["UID"])
        features["ITEM"] = self._item_lookup(features["ITEM"])
        features["CATEGORY"] = self._category_lookup(features["CATEGORY"])
        features["HISTORY_ITEM"] = self._item_lookup(features["HISTORY_ITEM"])
        features["HISTORY_CATEGORY"] = self._category_lookup(
            features["HISTORY_CATEGORY"]
        )
        features["NOCLK_HISTORY_ITEM"] = self._item_lookup(
            features["NOCLK_HISTORY_ITEM"]
        )
        features["NOCLK_HISTORY_CATEGORY"] = self._category_lookup(
            features["NOCLK_HISTORY_CATEGORY"]
        )

        tf.debugging.assert_equal(
            features["HISTORY_ITEM"].row_lengths(),
            features["HISTORY_CATEGORY"].row_lengths(),
        )

        features["SEQ_LENGTH"] = features["HISTORY_ITEM"].row_lengths()

    def __call__(self, features):
        self.forward(features)
        return features

    def vocabulary_sizes(self):
        return {
            "UID": self._uid_lookup.vocabulary_size(),
            "ITEM": self._item_lookup.vocabulary_size(),
            "CATEGORY": self._category_lookup.vocabulary_size(),
        }


class DNN(torch.nn.Module):
    def __init__(self, hidden_units, use_bn=False, activate_last=False):
        super(DNN, self).__init__()
        self._layers = torch.nn.ModuleList()
        for i in range(len(hidden_units) - 1):
            self._layers.append(torch.nn.Linear(hidden_units[i], hidden_units[i + 1]))
            if i == len(hidden_units) - 2 and activate_last:
                self._layers.append(torch.nn.Sigmoid())
                if use_bn:
                    self._layers.append(torch.nn.BatchNorm1d(hidden_units[i + 1]))

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class AuxiliaryNet(torch.nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self._bn1 = torch.nn.BatchNorm1d(input_size)
        self._dnn = DNN([input_size, 100, 50, 2])

    def forward(self, x):
        orig_shape = x.shape[0], x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        x = self._bn1(x)
        y = self._dnn(x)
        y = y.reshape(orig_shape[0], orig_shape[1], y.shape[-1])
        y_hat = torch.softmax(y, dim=1)
        return y_hat


class AuxiliaryLoss(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self._auxiliary_net = AuxiliaryNet(input_size)

    def forward(self, h_states, click_seq, noclick_seq, mask):
        click_input = torch.cat([h_states, click_seq], dim=-1)
        noclick_input = torch.cat([h_states, noclick_seq], dim=-1)

        click_prop = self._auxiliary_net(click_input)[:, :, 0]
        noclick_prop = self._auxiliary_net(noclick_input)[:, :, 0]

        click_loss = -torch.log(click_prop).reshape(-1, click_seq.shape[1]) * mask
        noclick_loss = (
            -torch.log(1 - noclick_prop).reshape(-1, noclick_seq.shape[1]) * mask
        )
        loss = torch.mean(click_loss + noclick_loss)
        return loss


class Attention(torch.nn.Module):
    def __init__(self, query_size, fact_size):
        super().__init__()
        self._query_size = query_size
        self._fact_size = fact_size
        self._query_dense = torch.nn.Linear(query_size, fact_size)
        self._prelu = torch.nn.PReLU(init=0.1)
        self._d_layer_1 = torch.nn.Linear(fact_size * 4, 80)
        self._d_layer_2 = torch.nn.Linear(80, 40)
        self._d_layer_3 = torch.nn.Linear(40, 1)

    def forward(self, query, facts, mask):
        if isinstance(facts, tuple):
            facts = torch.cat(facts, 2)
        if facts.dim == 2:
            facts = facts.unsqueeze(1)

        query = self._query_dense(query)
        query = self._prelu(query)
        queries = torch.tile(query, [1, facts.shape[1]]).reshape(facts.shape)
        din_all = torch.cat([queries, facts, queries - facts, queries * facts], dim=-1)
        d_layer_1_all = torch.sigmoid(self._d_layer_1(din_all))
        d_layer_2_all = torch.sigmoid(self._d_layer_2(d_layer_1_all))
        d_layer_3_all = self._d_layer_3(d_layer_2_all).reshape(-1, 1, facts.shape[1])
        scores = d_layer_3_all

        # Mask
        key_masks = mask.unsqueeze(1)
        paddings = torch.ones_like(scores) * (-(2**32) + 1)
        scores = torch.where(key_masks, scores, paddings)

        # Activation
        scores = F.softmax(scores, dim=-1)

        scores = scores.reshape(-1, facts.shape[1])
        output = (facts * scores.unsqueeze(-1)).reshape_as(facts)

        return output, scores


class TopFcLayer(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self._bn1 = torch.nn.BatchNorm1d(input_size)
        self._dnn1 = torch.nn.Linear(input_size, 200)
        self._dnn2 = torch.nn.Linear(200, 80)
        self._dnn3 = torch.nn.Linear(80, 2)
        self._logits = torch.nn.Linear(2, 1)

    def forward(self, x):
        bn1 = self._bn1(x)
        dnn1 = self._dnn1(bn1)
        dnn1 = torch.relu(dnn1)
        dnn2 = self._dnn2(dnn1)
        dnn2 = torch.relu(dnn2)
        dnn3 = self._dnn3(dnn2)
        logits = self._logits(dnn3)

        return logits


class VecAttGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._gate_linear = torch.nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self._candidate_linear = torch.nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, state, att_score):
        value = torch.sigmoid(self._gate_linear(torch.cat([input, state], dim=1)))
        r, u = torch.split(value, self._hidden_size, dim=1)

        r_state = r * state

        c = torch.tanh(self._candidate_linear(torch.cat([input, r_state], dim=1)))
        u = (1.0 - att_score) * u
        new_h = u * state + (1.0 - u) * c
        return new_h


class VecAttGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, gru_cell=None):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        if gru_cell is None:
            gru_cell = VecAttGRUCell(input_size, hidden_size)
        self._gru_cell = gru_cell
        self._batch_first = batch_first

    def forward(
        self, inputs: torch.Tensor, att_scores: torch.Tensor, seq_lengths: torch.Tensor
    ):
        if self._batch_first:
            inputs = inputs.transpose(0, 1)
            att_scores = att_scores.transpose(0, 1)

        max_seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]

        hx = torch.zeros(
            batch_size,
            self._hidden_size,
            dtype=inputs.dtype,
            device=inputs.device,
        )

        outputs = torch.zeros(
            max_seq_length,
            batch_size,
            self._hidden_size,
            dtype=inputs.dtype,
            device=inputs.device,
        )

        time = 0
        max_seq_length_actual = int(seq_lengths.max())
        for time in range(max_seq_length_actual):
            mask = seq_lengths > time
            new_hx = self._gru_cell(
                inputs[time, mask, ...],
                hx[mask, ...],
                att_scores[time, mask, ...],
            )
            outputs[time, mask, ...] = new_hx
            hx[mask, ...] = new_hx

        if self._batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, hx


class DIEN(torch.nn.Module):
    def __init__(self, vocab_sizes):
        super().__init__()
        self._vocab_sizes = vocab_sizes
        self._embeddings = self._get_embedding_tables()
        self._gru1 = torch.nn.GRU(
            input_size=ITEM_ALL_EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )
        self._auxiliary_loss = AuxiliaryLoss(
            input_size=HIDDEN_SIZE + ITEM_ALL_EMBEDDING_DIM
        )
        self._attention = Attention(
            query_size=ITEM_ALL_EMBEDDING_DIM,
            fact_size=HIDDEN_SIZE,
        )
        self._gru2 = VecAttGRU(
            input_size=HIDDEN_SIZE,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )
        self._top_fc_layer = TopFcLayer(
            input_size=EMBEDDING_DIM
            + ITEM_ALL_EMBEDDING_DIM
            * 3  # 3: item_emb, item_his_eb_sum, item_emb * item_his_eb_sum
            + HIDDEN_SIZE
        )

    def _get_embedding_tables(self):
        embeddings = {}
        embeddings["UID"] = torch.nn.Embedding(
            self._vocab_sizes["UID"] + 1, EMBEDDING_DIM, padding_idx=0
        )
        embeddings["ITEM"] = torch.nn.Embedding(
            self._vocab_sizes["ITEM"] + 1, EMBEDDING_DIM, padding_idx=0
        )
        embeddings["CATEGORY"] = torch.nn.Embedding(
            self._vocab_sizes["CATEGORY"] + 1, EMBEDDING_DIM, padding_idx=0
        )
        embeddings["HISTORY_ITEM"] = embeddings["ITEM"]
        embeddings["HISTORY_CATEGORY"] = embeddings["CATEGORY"]
        embeddings["NOCLK_HISTORY_ITEM"] = embeddings["ITEM"]
        embeddings["NOCLK_HISTORY_CATEGORY"] = embeddings["CATEGORY"]
        return torch.nn.ModuleDict(embeddings)

    def forward(self, features):
        uid_emb = self._embeddings["UID"](features["UID"])
        item_emb = torch.cat(
            [
                self._embeddings["ITEM"](features["ITEM"]),
                self._embeddings["CATEGORY"](features["CATEGORY"]),
            ],
            dim=1,
        )
        his_item_emb = torch.cat(
            [
                self._embeddings["HISTORY_ITEM"](features["HISTORY_ITEM"]),
                self._embeddings["HISTORY_CATEGORY"](features["HISTORY_CATEGORY"]),
            ],
            dim=2,
        )
        noclk_his_item_emb = torch.cat(
            [
                self._embeddings["NOCLK_HISTORY_ITEM"](features["NOCLK_HISTORY_ITEM"]),
                self._embeddings["NOCLK_HISTORY_CATEGORY"](
                    features["NOCLK_HISTORY_CATEGORY"]
                ),
            ],
            dim=2,
        )

        seq_length = features["SEQ_LENGTH"]
        item_his_eb_sum = torch.sum(his_item_emb, 1)

        # RNN layer_1
        packed_his_item_emb = torch.nn.utils.rnn.pack_padded_sequence(
            his_item_emb, seq_length, batch_first=True, enforce_sorted=False
        )
        packed_rnn_output_1, _ = self._gru1(packed_his_item_emb)
        rnn_output_1, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_rnn_output_1,
            batch_first=True,
            padding_value=0.0,
            total_length=MAX_SEQ_LENGTH,
        )
        mask = torch.arange(his_item_emb.size(1))[None, :] < seq_length[:, None]

        # Aux loss
        aux_loss = self._auxiliary_loss(
            rnn_output_1[:, :-1, :],
            his_item_emb[:, 1:, :],
            noclk_his_item_emb[:, 1:, :],
            mask[:, 1:].float(),
        )

        # Attention layer
        _, alphas = self._attention(item_emb, rnn_output_1, mask)

        # RNN layer_2
        _, final_state2 = self._gru2(rnn_output_1, alphas.unsqueeze(-1), seq_length)

        # Prepare input for the top MLP layer
        top_input = torch.cat(
            [
                uid_emb,
                item_emb,
                item_his_eb_sum,
                item_emb * item_his_eb_sum,
                final_state2,
            ],
            dim=1,
        )

        # Top MLP layer
        logits = self._top_fc_layer(top_input)

        prob = torch.sigmoid(logits).flatten()

        return prob, aux_loss


def tensorflow_to_pytorch(obj) -> torch.Tensor | Mapping | Sequence:
    if isinstance(obj, tf.RaggedTensor):
        return torch.from_numpy(obj.to_tensor().numpy())
        # FIXME: converting a ragged tensor into dense tensor introduces extra items in history columns,
        #        which makes the input contains more items corresponding to hash value of 0 and screws
        #        the input.
    elif tf.is_tensor(obj):
        return torch.from_numpy(obj.numpy())
    elif isinstance(obj, Mapping):
        return {key: tensorflow_to_pytorch(value) for key, value in obj.items()}
    elif isinstance(obj, Sequence):
        return [tensorflow_to_pytorch(item) for item in obj]
    else:
        return obj


def main():
    # check dataset and count data set size
    print("Checking dataset...")
    train_file = args.data_location + "/local_train_splitByUser"
    test_file = args.data_location + "/local_test_splitByUser"
    if (
        (not os.path.exists(train_file))
        or (not os.path.exists(test_file))
        or (not os.path.exists(train_file + "_neg"))
        or (not os.path.exists(test_file + "_neg"))
    ):
        print("Dataset does not exist in the given data_location.")
        sys.exit()
    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of training dataset is {}".format(no_of_training_examples))
    print("Numbers of test dataset is {}".format(no_of_test_examples))

    # set batch size, eporch & steps
    batch_size = args.batch_size

    if args.steps == 0:
        no_of_epochs = 1
        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size
        )
    else:
        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples
        )
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)
    print("The training steps is {}".format(train_steps))
    print("The testing steps is {}".format(test_steps))

    # set fixed random seed
    tf.random.set_seed(args.seed)
    torch.manual_seed(args.seed)

    # create data pipline of train & test dataset
    train_dataset = build_model_input(train_file, batch_size, no_of_epochs)
    test_dataset = build_model_input(test_file, batch_size, 1)

    preprocessor = FeaturePreprocessor(args.data_location)
    model = DIEN(preprocessor.vocabulary_sizes())
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    time_start = time.perf_counter()
    with tqdm(total=train_steps, desc=f"", unit="it") as pbar:
        start_time = time.perf_counter()
        for step, (features, label) in islice(enumerate(train_dataset), train_steps):
            features = tensorflow_to_pytorch(preprocessor(features))
            label = tensorflow_to_pytorch(label).float()

            # Forward
            pred, aux_loss = model(features)
            crt_loss = loss_fn(pred, label)
            loss = crt_loss + aux_loss

            # Backward
            model.zero_grad()
            loss.backward()

            # Optmizer step
            optimizer.step()

            pbar.update(1)
            if (step + 1) % 100 == 0:
                elapsed_time = time.perf_counter() - start_time
                tqdm.write(
                    f"loss = {loss.detach().numpy():.8f}, "
                    f"aux_loss = {aux_loss.detach().numpy():.8f}, "
                    f"steps = {step+1}/{train_steps} ({elapsed_time:.3f} sec)"
                )
                start_time = time.perf_counter()  # Reset the start time
    time_end = time.perf_counter()
    print("Training completed.")
    time_cost = time_end - time_start
    global global_time_cost
    global_time_cost = time_cost

    os.makedirs(result_dir, exist_ok=True)
    print(global_auc)
    print(global_time_cost)
    with open(result_path, "w") as f:
        f.write(str(global_time_cost) + "\n")
        f.write(str(global_auc) + "\n")


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


# Get parse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_location",
        help="Full path of train data",
        required=False,
        default="./data",
    )
    parser.add_argument(
        "--steps", help="set the number of steps on train dataset", type=int, default=0
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size to train. Default is 512",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--seed", help="set the random seed for tensorflow", type=int, default=2021
    )
    parser.add_argument(
        "--learning_rate",
        help="Learning rate for deep model",
        type=float,
        default=0.001,
    )

    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    main()
