import csv
import numpy as np
import pandas as pd
from dataclasses import dataclass

ENCODED_DATA_FILEPATH = 'data/encoded_data.csv'
MIN_MAX_LABEL_FILEPATH = 'data/label_min_max.csv'

@dataclass
class QueryCardinalityDataset:
    data: np.ndarray
    labels_min_max_values: np.ndarray

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def extract_data(file_name, num_materialized_samples):
    joins = []
    predicates = []
    tables = []
    samples = []
    label = []

    # Load queries
    with open(file_name + ".csv", 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            label.append(row[3])
    print("Loaded queries")

    # Load bitmaps
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(file_name + ".bitmaps", 'rb') as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")

    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]

    return joins, predicates, tables, samples, label

def get_one_hot_encoding(data, item2idx):
    one_hot_encoding = np.zeros(len(item2idx.keys()))
    
    for item in data:
        one_hot_encoding[item2idx[item]] = 1

    return one_hot_encoding

def item2index(item_list):
    return {item:idx for idx, item in enumerate(sorted(item_list))}

def encode_one_hot(data):
    unique_items = []
    for d in data:
        unique_items.extend(d)

    # Remove duplicates
    unique_items = set(unique_items)

    item2idx = item2index(unique_items)

    return [get_one_hot_encoding(d, item2idx) for d in data], item2idx

def min_max_value_encoder(column, value, column_statistics):
    stats = column_statistics.loc[column_statistics['name'] == column, :]
    return (value - stats['min'])/(stats['max'] - stats['min'])

def enconde_predicates(predicates, column_statistics):
    columns = {p[0] for pred in predicates for p in pred if len(p) == 3}
    columns = sorted(columns)
    cols2idx = item2index(columns)

    operators = {p[1] for pred in predicates for p in pred if len(p) == 3}
    operators = sorted(operators)
    operator2idx = item2index(operators)
    
    #TODO: Get rid of one operator (< or >) by using 1 - value ---> Thanks to Jens :D

    # predicates have size = #one_hot_operators + 1 position for the predicate value
    chunck_len_predicate = len(operators) + 1
    vector_len_predicate = len(columns) * chunck_len_predicate
    
    predicate_encoding = []
    for pred in predicates:
        vec = np.zeros(vector_len_predicate)
        for p in pred:
            if len(p) == 3:
                col, op, val = p
                chunck = cols2idx[col]
                operator_position_in_vec = chunck * chunck_len_predicate + operator2idx[op]
                vec[operator_position_in_vec] = 1
                vec[operator_position_in_vec+1] = min_max_value_encoder(col, float(val), column_statistics)
                predicate_encoding.append(vec)
    
    return predicate_encoding

def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(l)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val

def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype=np.int64)

def transform_and_encode_data(num_queries=10000, num_materialized_samples=1000):
    file_name_queries = "data/train"
    file_name_column_min_max_vals = "data/column_min_max_vals.csv"

    column_statistics = pd.read_csv(file_name_column_min_max_vals)
    
    joins, predicates, tables, samples, label = extract_data(file_name_queries, num_materialized_samples)
    
    label_normalized, min_label_val, max_label_val = normalize_labels(label)

    table_encodings, table2idx = encode_one_hot(tables)
    join_encodings, join2idx = encode_one_hot(joins)
    predicate_encodings = enconde_predicates(predicates, column_statistics)

    vector_size = len(table2idx.keys()) + len(join2idx.keys()) + predicate_encodings[0].shape[0] + 1

    dataset = []
    for t,j,p,l in zip(table_encodings, join_encodings, predicate_encodings, label_normalized):
        vec = np.hstack((t, j))
        vec = np.hstack((vec, p))
        vec = np.hstack((vec, l))
        assert vec.shape[0] == vector_size, f"vec has size {vec.shape[0]}, but should be {vector_size}"
        dataset.append(vec)

    np.savetxt(ENCODED_DATA_FILEPATH, np.array(dataset), delimiter=',')
    np.savetxt(MIN_MAX_LABEL_FILEPATH, np.array([min_label_val, max_label_val]), delimiter=',')

    return dataset

def load_dataset():
    data = np.loadtxt(ENCODED_DATA_FILEPATH, delimiter=',')
    labels_min_max_values = np.loadtxt(MIN_MAX_LABEL_FILEPATH, delimiter=',')
    
    return QueryCardinalityDataset(data, labels_min_max_values)

def main():
    transform_and_encode_data()

if __name__ == '__main__':
    main()
