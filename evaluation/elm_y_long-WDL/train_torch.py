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
from deepctr_torch.layers import activation_layer
from itertools import islice

logger = logging.getLogger(__name__)

from tensorflow.python.feature_column import utils as fc_utils

result_dir = "/tmp/tianchi/result/WDL/"
result_path = result_dir + "result"
global_time_cost = 0
global_auc = 0

# Definition of some constants
LABEL_COLUMN = ["clicked"]
LABEL_COLUMN_DEFAULTS = [0]
USER_COLUMNS = [
    "user_id",
]
USER_COLUMNS_DEFAULTS = [""]
ITEM_COLUMN = [
    "shop_id",
    "item_id",
    "district_id",
    "shop_aoi_id",
    "shop_geohash_6",
    "brand_id",
    "category_1_id",
    "merge_standard_food_id",
    "rank_7",
    "rank_30",
    "rank_90",
]
ITEM_COLUMN_DEFAULTS = ["", "", "0", "", "", "0", "0", "0", 0, 0, 0]
HISTORY_COLUMN = [
    "shop_id_list",
    "item_id_list",
    "category_1_id_list",
    "merge_standard_food_id_list",
    "brand_id_list",
    "price_list",
    "shop_aoi_id_list",
    "shop_geohash6_list",
    "timediff_list",
    "hours_list",
    "time_type_list",
    "weekdays_list",
]
HISTORY_COLUMN_DEFAULTS = ["", "", "", "", "", "0", "", "", "0", "-0", "", ""]
USER_TZ_COLUMN = ["times", "hours", "time_type", "weekdays"]
USER_TZ_COLUMN_DEFAULTS = ["0", 0, "", 0]
DEFAULTS = (
    LABEL_COLUMN_DEFAULTS
    + USER_COLUMNS_DEFAULTS
    + ITEM_COLUMN_DEFAULTS
    + HISTORY_COLUMN_DEFAULTS
    + USER_TZ_COLUMN_DEFAULTS
)

FEATURE_COLUMNS = USER_COLUMNS + ITEM_COLUMN + HISTORY_COLUMN + USER_TZ_COLUMN
TRAIN_DATA_COLUMNS = LABEL_COLUMN + FEATURE_COLUMNS
SHARE_EMBEDDING_COLS = [
    ["shop_id", "shop_id_list"],
    ["item_id", "item_id_list"],
    ["category_1_id", "category_1_id_list"],
    ["merge_standard_food_id", "merge_standard_food_id_list"],
    ["brand_id", "brand_id_list"],
    ["shop_aoi_id", "shop_aoi_id_list"],
    ["shop_geohash_6", "shop_geohash6_list"],
]
EMBEDDING_COLS = ["user_id", "district_id", "times", "timediff_list"]
CONTINUOUS_COLUMNS = [
    "rank_7",
    "rank_30",
    "rank_90",
    "hours",
]
CONTINUOUS_HISTORY_COLUMNS = ["price_list", "hours_list"]
TYPE_COLS = ["time_type", "time_type_list"]
TYPE_LIST = ["lunch", "night", "dinner", "tea", "breakfast"]

HASH_BUCKET_SIZES = 100000
EMBEDDING_DIMENSIONS = 16

DEEP_COLUMNS = []
WIDE_COLUMNS = []

for columns in SHARE_EMBEDDING_COLS:
    for col in columns:
        DEEP_COLUMNS.append(col + "_emb")
        if col not in HISTORY_COLUMN:
            WIDE_COLUMNS.append(col)
for column in EMBEDDING_COLS:
    DEEP_COLUMNS.append(column + "_emb")
    WIDE_COLUMNS.append(column)
for column in CONTINUOUS_COLUMNS + CONTINUOUS_HISTORY_COLUMNS:
    DEEP_COLUMNS.append(column)
    WIDE_COLUMNS.append(column)
for column in TYPE_COLS:
    DEEP_COLUMNS.append(column + "_emb")
    WIDE_COLUMNS.append(column)


# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs):
    def parse_csv(value):
        logger.info("Parsing {}".format(filename))
        column_headers = TRAIN_DATA_COLUMNS
        columns = tf.io.decode_csv(value, record_defaults=DEFAULTS)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        return features, labels

    files = filename
    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(files)
    dataset = dataset.shuffle(
        buffer_size=20000, seed=args.seed
    )  # fix seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(2)
    return dataset


def preprocess_features(features):
    for key in HISTORY_COLUMN:
        features[key] = tf.strings.split(features[key], ";")

    for key in CONTINUOUS_HISTORY_COLUMNS:
        length = features[key].row_lengths()
        length = tf.expand_dims(length, -1)
        features[key] = features[key].to_tensor(default_value="0")
        features[key] = tf.strings.to_number(features[key])

        features[key] = tf.reduce_sum(features[key], axis=1, keepdims=True)
        features[key] = tf.math.divide(features[key], tf.cast(length, tf.float32))

    processed = {}

    for columns in SHARE_EMBEDDING_COLS:
        for col in columns:
            hashed = (
                keras.layers.Hashing(
                    num_bins=HASH_BUCKET_SIZES - 1,
                    output_mode="int",
                )(features[col])
                + 1
            )
            processed[col] = hashed

    for column in EMBEDDING_COLS:
        hashed = (
            keras.layers.Hashing(
                num_bins=HASH_BUCKET_SIZES - 1,
                output_mode="int",
            )(features[column])
            + 1
        )
        processed[column] = hashed

    for column in CONTINUOUS_COLUMNS + CONTINUOUS_HISTORY_COLUMNS:
        processed[column] = features[column]

    for column in TYPE_COLS:
        hashed = keras.layers.StringLookup(
            vocabulary=TYPE_LIST,
            output_mode="int",
        )(features[column])
        processed[column] = hashed

    return processed


class DNN(torch.nn.Module):
    def __init__(self, hidden_units):
        super(DNN, self).__init__()
        self._layers = torch.nn.ModuleList()
        for i in range(len(hidden_units) - 1):
            self._layers.append(torch.nn.Linear(hidden_units[i], hidden_units[i + 1]))
            if hidden_units[i + 1] != 1:
                torch.nn.init.xavier_uniform_(self._layers[-1].weight)
                self._layers.append(torch.nn.ReLU())

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class SparseLinear(torch.nn.Module):
    def __init__(self):
        super(SparseLinear, self).__init__()
        self._linear_embeddings = torch.nn.ModuleDict(
            {
                name: torch.nn.Embedding(len(TYPE_LIST) + 1, 1, padding_idx=0)
                if name in TYPE_COLS
                else torch.nn.Embedding(HASH_BUCKET_SIZES, 1, padding_idx=0)
                for name in WIDE_COLUMNS
                if name not in (CONTINUOUS_COLUMNS + CONTINUOUS_HISTORY_COLUMNS)
            }
        )

    def forward(self, features):
        linear_inputs = []
        for name in WIDE_COLUMNS:
            if name in self._linear_embeddings:
                if name in HISTORY_COLUMN:
                    linear_inputs.append(
                        torch.sum(self._linear_embeddings[name](features[name]), dim=1)
                    )
                else:
                    linear_inputs.append(self._linear_embeddings[name](features[name]))
            else:
                linear_inputs.append(
                    features[name].reshape(features[name].shape[0], -1)
                )

        linear_input = torch.cat(linear_inputs, dim=1).sum(dim=1, keepdim=True)
        linear_logit = torch.sum(linear_input, dim=1, keepdim=True)

        return linear_logit


class WDL(torch.nn.Module):
    def __init__(self):
        super(WDL, self).__init__()
        self._embeddings = self._get_embedding_tables()
        self._dnn = DNN(
            [326, 1024, 512, 256, 64, 1]
        )
        self._linear = SparseLinear()

    def _get_embedding_tables(self):
        embeddings = {}

        for columns in SHARE_EMBEDDING_COLS:
            emb = torch.nn.Embedding(
                HASH_BUCKET_SIZES,
                EMBEDDING_DIMENSIONS,
                padding_idx=0,
            )
            for col in columns:
                embeddings[col] = emb
        for column in EMBEDDING_COLS:
            embeddings[column] = torch.nn.Embedding(
                HASH_BUCKET_SIZES,
                EMBEDDING_DIMENSIONS,
                padding_idx=0,
            )
        for column in TYPE_COLS:
            embeddings[column] = torch.nn.Embedding(
                len(TYPE_LIST) + 1,  # An extra token for OOV words
                EMBEDDING_DIMENSIONS,
                padding_idx=0,
            )

        return torch.nn.ModuleDict(embeddings)

    def forward(self, features):
        for k, emb_fn in self._embeddings.items():
            emb = emb_fn(features[k])
            if k in HISTORY_COLUMN:
                emb = torch.sum(emb, dim=1)
            features[k + "_emb"] = emb

        dnn_input = torch.cat(
            [features[col].reshape(features[col].shape[0], -1) for col in DEEP_COLUMNS],
            dim=1,
        )

        dnn_input = F.normalize(dnn_input, p=2, dim=1)
        dnn_output = self._dnn(dnn_input)

        linear_output = self._linear(features)

        logits = dnn_output + linear_output
        prob = torch.sigmoid(logits).flatten()

        return prob


def tensorflow_to_pytorch(obj) -> torch.Tensor | Mapping | Sequence:
    if isinstance(obj, tf.RaggedTensor):
        return torch.from_numpy(obj.to_tensor().numpy())
    elif tf.is_tensor(obj):
        return torch.from_numpy(obj.numpy())
    elif isinstance(obj, Mapping):
        return {key: tensorflow_to_pytorch(value) for key, value in obj.items()}
    elif isinstance(obj, Sequence):
        return [tensorflow_to_pytorch(item) for item in obj]
    else:
        return obj


def main():
    # check dataset
    print("Checking dataset...")
    train_file = os.path.join(args.data_location, "train.hash8.enlarged8.csv")
    test_file = os.path.join(args.data_location, "eval.hash8.csv")
    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        print("Dataset does not exist in the given data_location.")
        sys.exit()
    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))

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

    # set directory path
    # model_dir = os.path.join(args.output_dir, "model_DeepFM_" + str(int(time.time())))
    # checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    # print("Saving model checkpoints to " + checkpoint_dir)

    # create data pipline of train & test dataset
    train_dataset = build_model_input(train_file, batch_size, no_of_epochs)
    test_dataset = build_model_input(test_file, batch_size, 1)

    model = WDL()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    time_start = time.perf_counter()
    with tqdm(total=train_steps, desc=f"", unit="it") as pbar:
        start_time = time.perf_counter()
        for step, (features, label) in islice(enumerate(train_dataset), train_steps):
            features = tensorflow_to_pytorch(preprocess_features(features))
            label = tensorflow_to_pytorch(label).float()

            # Forward
            pred = model(features)
            loss = loss_fn(pred, label)

            # Backward
            model.zero_grad()
            loss.backward()

            # Optmizer step
            optimizer.step()

            pbar.update(1)
            if (step + 1) % 100 == 0:
                elapsed_time = time.perf_counter() - start_time
                tqdm.write(
                    f"loss = {loss.detach().numpy():.8f}, steps = {step+1}/{train_steps} ({elapsed_time:.3f} sec)"
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
        default=0.01,
    )
    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    main()
