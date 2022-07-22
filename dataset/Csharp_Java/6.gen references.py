import tensorflow as tf


keys = ['cs_t', 'java_t']

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

raw_data = tf.data.TFRecordDataset(r"./dataset/test.tfrecords").map(_parse_data)
keys_data = [[], []]

for each_data in iter(raw_data):
    def record_each(index, data):
        data = data.numpy().tolist()
        data = [[bytes.decode(word) for word in line if len(word) != 0] for line in data]
        data = [''.join(line).replace("$$", "").replace('▁', " ").strip() for line in data if len(line) != 0]
        keys_data[index].append(' '.join(data))

    record_each(0, each_data[0])
    record_each(1, each_data[1])


def write_reference(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))


write_reference(keys[0]+'_references.txt', keys_data[0])
write_reference(keys[1]+'_references.txt', keys_data[1])
