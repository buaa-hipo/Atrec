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

import time
import argparse
import os
import math
import logging
import tensorflow as tf
from collections.abc import Mapping, Sequence
from itertools import islice


class Timer:
    def __init__(self, message="Execution time"):
        self.message = message  # Custom message to be printed with the execution time

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        tqdm.write(f"{self.message:10}: {self.elapsed:.7f} seconds")


logger = logging.getLogger(__name__)

result_dir = "/tmp/tianchi/result/DIEN/"
result_path = result_dir + "result"
global_time_cost = 0
global_auc = 0

EMBEDDING_DIM = 4
ITEM_ALL_EMBEDDING_DIM = EMBEDDING_DIM * 6
HIDDEN_SIZE = EMBEDDING_DIM * 2
ATTENTION_SIZE = EMBEDDING_DIM * 2
ITEM_PACKSIZE = 5000


# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs):
    def parse_npz(values):
        logger.info("Parsing {}".format(filename))

        feature_map = {
            "history_item_array": tf.io.FixedLenFeature((100), tf.int64),
            "neg_history_item_array": tf.io.FixedLenFeature((100), tf.int64),
            "history_item_array": tf.io.FixedLenFeature((100), tf.int64),
            "neg_history_item_array": tf.io.FixedLenFeature((100), tf.int64),
            "history_cate_array": tf.io.FixedLenFeature((100), tf.int64),
            "neg_history_cate_array": tf.io.FixedLenFeature((100), tf.int64),
            "history_shop_array": tf.io.FixedLenFeature((100), tf.int64),
            "neg_history_shop_array": tf.io.FixedLenFeature((100), tf.int64),
            "history_node_array": tf.io.FixedLenFeature((100), tf.int64),
            "neg_history_node_array": tf.io.FixedLenFeature((100), tf.int64),
            "history_product_array": tf.io.FixedLenFeature((100), tf.int64),
            "neg_history_product_array": tf.io.FixedLenFeature((100), tf.int64),
            "history_brand_array": tf.io.FixedLenFeature((100), tf.int64),
            "neg_history_brand_array": tf.io.FixedLenFeature((100), tf.int64),
            "target_array": tf.io.FixedLenFeature((2), tf.int64),
            "source_array": tf.io.FixedLenFeature((7), tf.int64),
        }
        all_columns = tf.io.parse_example(values, features=feature_map)
        cond = tf.math.greater(all_columns["history_item_array"], 0)
        all_columns["history_mask"] = tf.cast(cond, tf.float32)
        source_array = all_columns.pop("source_array")
        uid, item, cate, shop, node, product, brand = tf.split(
            source_array, [1, 1, 1, 1, 1, 1, 1], axis=1
        )
        all_columns["uid"] = tf.reshape(uid, [-1])
        all_columns["item"] = tf.reshape(item, [-1])
        all_columns["cate"] = tf.reshape(cate, [-1])
        all_columns["shop"] = tf.reshape(shop, [-1])
        all_columns["node"] = tf.reshape(node, [-1])
        all_columns["product"] = tf.reshape(product, [-1])
        all_columns["brand"] = tf.reshape(brand, [-1])

        labels = all_columns.pop("target_array")[:, 0]
        features = all_columns

        return features, labels

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.shuffle(
        buffer_size=20000, seed=args.seed
    )  # set seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_npz, num_parallel_calls=1)
    dataset = dataset.prefetch(2)
    return dataset


class FeaturePreprocessor:
    def __init__(self):
        pass

    def forward(self, features):
        return features

    def __call__(self, features):
        self.forward(features)
        return features

    def vocabulary_sizes(self):
        return {
            "uid": 7956430,
            "item": 34196611,
            "cate": 5596,
            "shop": 4377722,
            "node": 2975349,
            "product": 65624,
            "brand": 584181,
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
            input_size=ITEM_ALL_EMBEDDING_DIM * 3 + HIDDEN_SIZE
        )

    def _get_embedding_tables(self):
        embeddings = {}
        for feature, size in self._vocab_sizes.items():
            embeddings[feature] = torch.nn.Embedding(
                size + 1, EMBEDDING_DIM, padding_idx=0
            )
        return torch.nn.ModuleDict(embeddings)

    def _embedding_input_layer(self, features):
        for k in features:
            features[k] = torch.where(
                features[k] > 0, features[k], torch.zeros_like(features[k])
            )

        item_id_his_batch = features["history_item_array"]
        cate_his_batch = features["history_cate_array"]
        shop_his_batch = features["history_shop_array"]
        node_his_batch = features["history_node_array"]
        product_his_batch = features["history_product_array"]
        brand_his_batch = features["history_brand_array"]
        item_id_batch = features["item"]
        cate_id_batch = features["cate"]
        shop_id_batch = features["shop"]
        node_id_batch = features["node"]
        product_id_batch = features["product"]
        brand_id_batch = features["brand"]
        mask = features["history_mask"]

        item_id_neg_batch = features["neg_history_item_array"]
        cate_neg_batch = features["neg_history_cate_array"]
        shop_neg_batch = features["neg_history_shop_array"]
        node_neg_batch = features["neg_history_node_array"]
        product_neg_batch = features["neg_history_product_array"]
        brand_neg_batch = features["neg_history_brand_array"]

        seq_length = item_id_his_batch.shape[1]

        item_id_batch_emb = self._embeddings["item"](item_id_batch)
        item_id_his_batch_emb = self._embeddings["item"](item_id_his_batch)
        cate_id_batch_emb = self._embeddings["cate"](cate_id_batch)
        cate_his_batch_emb = self._embeddings["cate"](cate_his_batch)
        shop_id_batch_emb = self._embeddings["shop"](shop_id_batch)
        shop_his_batch_emb = self._embeddings["shop"](shop_his_batch)
        node_id_batch_emb = self._embeddings["node"](node_id_batch)
        node_his_batch_emb = self._embeddings["node"](node_his_batch)
        product_id_batch_emb = self._embeddings["product"](product_id_batch)
        product_his_batch_emb = self._embeddings["product"](product_his_batch)
        brand_id_batch_emb = self._embeddings["brand"](brand_id_batch)
        brand_his_batch_emb = self._embeddings["brand"](brand_his_batch)

        neg_item_his_emb = self._embeddings["item"](item_id_neg_batch)
        neg_cate_his_emb = self._embeddings["cate"](cate_neg_batch)
        neg_shop_his_emb = self._embeddings["shop"](shop_neg_batch)
        neg_node_his_emb = self._embeddings["node"](node_neg_batch)
        neg_product_his_emb = self._embeddings["product"](product_neg_batch)
        neg_brand_his_emb = self._embeddings["brand"](brand_neg_batch)

        neg_his_eb = torch.cat(
            [
                neg_item_his_emb,
                neg_cate_his_emb,
                neg_shop_his_emb,
                neg_node_his_emb,
                neg_product_his_emb,
                neg_brand_his_emb,
            ],
            dim=-1,
        ) * mask.unsqueeze(-1)

        item_eb = torch.cat(
            [
                item_id_batch_emb,
                cate_id_batch_emb,
                shop_id_batch_emb,
                node_id_batch_emb,
                product_id_batch_emb,
                brand_id_batch_emb,
            ],
            dim=-1,
        )

        item_his_eb = torch.cat(
            [
                item_id_his_batch_emb,
                cate_his_batch_emb,
                shop_his_batch_emb,
                node_his_batch_emb,
                product_his_batch_emb,
                brand_his_batch_emb,
            ],
            dim=-1,
        ) * mask.unsqueeze(-1)

        return item_eb, item_his_eb, neg_his_eb, mask, seq_length

    def forward(self, features):
        with Timer("emb"):
            (
                item_emb,
                his_item_emb,
                noclk_his_item_emb,
                mask,
                seq_length,
            ) = self._embedding_input_layer(features)

        seq_lengths = torch.full(
            (his_item_emb.shape[0],), seq_length, dtype=torch.int64
        )
        item_his_eb_sum = torch.sum(his_item_emb, 1)

        # RNN layer_1
        with Timer("gru1"):
            packed_his_item_emb = torch.nn.utils.rnn.pack_padded_sequence(
                his_item_emb, seq_lengths, batch_first=True, enforce_sorted=False
            )
            packed_rnn_output_1, _ = self._gru1(packed_his_item_emb)
            rnn_output_1, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_rnn_output_1,
                batch_first=True,
                padding_value=0.0,
                total_length=seq_length,
            )

        # Aux loss
        with Timer("aux"):
            aux_loss = self._auxiliary_loss(
                rnn_output_1[:, :-1, :],
                his_item_emb[:, 1:, :],
                noclk_his_item_emb[:, 1:, :],
                mask[:, 1:].float(),
            )

        # Attention layer
        with Timer("attn"):
            _, alphas = self._attention(item_emb, rnn_output_1, mask != 0)

        # RNN layer_2
        with Timer("gru2"):
            _, final_state2 = self._gru2(
                rnn_output_1, alphas.unsqueeze(-1), seq_lengths
            )

        with Timer("mlp"):
            # Prepare input for the top MLP layer
            top_input = torch.cat(
                [
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
    train_file = args.data_location + "/train_sample_0.tfrecords"
    test_file = args.data_location + "/test_sample_0.tfrecords"

    no_of_training_examples = sum(1 for _ in tf.data.TFRecordDataset(train_file))
    no_of_test_examples = sum(1 for _ in tf.data.TFRecordDataset(test_file))
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

    preprocessor = FeaturePreprocessor()
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
            with Timer("bce loss"):
                crt_loss = loss_fn(pred, label)
            loss = crt_loss + aux_loss

            # Backward
            model.zero_grad()
            with Timer("backward"):
                loss.backward()

            with Timer("optimizer"):
                # Optmizer step
                optimizer.step()

            tqdm.write("=" * 50)
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
