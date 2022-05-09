import csv
import enum
from matplotlib.pyplot import get
import numpy as np
import pandas as pd

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_data(file_name, num_materialized_samples):
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


def load_and_encode_train_data(num_queries=10000, num_materialized_samples=1000):
    file_name_queries = "data/train"
    file_name_column_min_max_vals = "data/column_min_max_vals.csv"

    joins, predicates, tables, samples, label = load_data(file_name_queries, num_materialized_samples)

    table_encodings, table2idx = encode_one_hot(tables)
    join_encodings, join2idx = encode_one_hot(joins)


def main():
    load_and_encode_train_data()

if __name__ == '__main__':
    main()
