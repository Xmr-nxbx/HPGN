import json
import os
import tensorflow as tf

keys = ['cs_t', 'java_t']

# 0: [0: 5)  1: [5, 10)  2: [10: 20)  3: [20: 30)  4: [30: ]
word_inline_counter = [[0] * 5 for _ in range(2)]
max_word_inline = [0, 0]
# 0: [0: 3)  1: [3, 5)  2: [5, 10)  3: [10, 15)  4: [15: ]
line_counter = [[0] * 5 for _ in range(2)]
max_line = [0, 0]
# 0: [0: 60)  1: [60: 130)  2: [130: 210)  3: [210: 300)  4: [300: ]
word_counter = [[0] * 5 for _ in range(2)]
max_word = [0]*2

word_sum = [0]*2
line_sum = [0]*2

module_path = os.path.dirname(__file__)

feature_description = {
    keys[0]: tf.io.VarLenFeature(tf.string),
    keys[1]: tf.io.VarLenFeature(tf.string),
}


def _parse_data(serialize_string):
    content_feature, sequence_feature = tf.io.parse_single_sequence_example(serialize_string,
                                                                            sequence_features=feature_description)
    # sparse tensor
    k1_data, k2_data = sequence_feature[keys[0]], sequence_feature[keys[1]]
    k1_data, k2_data = tf.sparse.to_dense(k1_data, b''), tf.sparse.to_dense(k2_data, b'')
    return k1_data, k2_data


raw_data1 = tf.data.TFRecordDataset(module_path + r"./dataset/train.tfrecords").map(_parse_data)
raw_data2 = tf.data.TFRecordDataset(module_path + r"./dataset/valid.tfrecords").map(_parse_data)
raw_data3 = tf.data.TFRecordDataset(module_path + r"./dataset/test.tfrecords").map(_parse_data)
raw_data = [raw_data1, raw_data2, raw_data3]

dataset_step = 0
dataset_line_step = [0] * 2
for data in raw_data:
    for datasets in iter(data):
        dataset_step += 1

        def count_func(index, data):
            data = data.numpy().tolist()
            data = [[word for word in line if len(word) != 0] for line in data]
            data = [line for line in data if len(line) != 0]
            line_len = len(data)
            dataset_line_step[index] += len(data)
            line_sum[index] += line_len
            if line_len > max_line[index]:
                max_line[index] = line_len
            if line_len >= 0 and line_len < 3:
                line_counter[index][0] += 1
            elif line_len >= 3 and line_len < 5:
                line_counter[index][1] += 1
            elif line_len >= 5 and line_len < 10:
                line_counter[index][2] += 1
            elif line_len >= 10 and line_len < 15:
                line_counter[index][3] += 1
            else:
                line_counter[index][4] += 1

            def count_each_line(line):
                word_inline_len = len(line)
                if word_inline_len > max_word_inline[index]:
                    max_word_inline[index] = word_inline_len
                if word_inline_len >= 0 and word_inline_len < 5:
                    word_inline_counter[index][0] += 1
                elif word_inline_len >= 5 and word_inline_len < 10:
                    word_inline_counter[index][1] += 1
                elif word_inline_len >= 10 and word_inline_len < 20:
                    word_inline_counter[index][2] += 1
                elif word_inline_len >= 20 and word_inline_len < 30:
                    word_inline_counter[index][3] += 1
                else:
                    word_inline_counter[index][4] += 1
            for line in data:
                count_each_line(line)
            word_len = sum([len(line) for line in data])
            word_sum[index] += word_len
            if word_len > max_word[index]:
                max_word[index] = word_len
            if word_len >= 0 and word_len < 60:
                word_counter[index][0] += 1
            elif word_len >= 60 and word_len < 130:
                word_counter[index][1] += 1
            elif word_len >= 130 and word_len < 200:
                word_counter[index][2] += 1
            elif word_len >= 200 and word_len < 300:
                word_counter[index][3] += 1
            else:
                word_counter[index][4] += 1

        count_func(0, datasets[0])
        count_func(1, datasets[1])

for i in range(len(keys)):
    print('\nkeys: %s, statistics' % keys[i])
    print('word:')
    print('max: %d  avg: %f  sum: %d  counter-all: %d  [0: 60): %d  [60: 130): %d  [130: 200): %d  [200: 300): %d  [300: ] : %d' % (max_word[i], word_sum[i]/dataset_step, word_sum[i], sum(word_counter[i]), *(word_counter[i])))
    print('line:')
    print('max: %d  avg: %f  sum: %d  counter-all: %d  [0: 3): %d  [3: 5): %d  [5: 10): %d  [10: 15): %d  [15: ] : %d' % (max_line[i], line_sum[i]/dataset_step, line_sum[i], sum(line_counter[i]), *(line_counter[i])))
    print('word inline:')
    print('max: %d  avg: %f           counter-all: %d  [0: 5): %d  [5: 10): %d  [10: 20): %d  [20: 30): %d  [30: ] : %d' % (max_word_inline[i], word_sum[i]/dataset_line_step[i], sum(word_inline_counter[i]), *(word_inline_counter[i])))

