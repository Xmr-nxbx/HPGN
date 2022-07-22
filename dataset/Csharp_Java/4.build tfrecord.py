import tensorflow as tf
import ijson
import os

keys = ['cs_t', 'java_t']
path = r'./dataset'


def write_to_file(from_file, to_file):
    with tf.io.TFRecordWriter(to_file) as writer:
        parser = ijson.parse(open(from_file, 'r', encoding='utf-8'))
        keys_data = [None] * len(keys)

        for prefix, event, value in parser:
            # print(prefix, event, value)
            prefixes = prefix.split('.')
            if len(prefixes) == 2 and prefixes[1] in keys and event == 'start_array':
                tmp = []
            elif len(prefixes) == 3 and prefixes[1] in keys and event == 'start_array':
                tmp.append([])
            elif len(prefixes) == 4 and prefixes[1] in keys:
                tmp[-1].append(str.encode(value))
            elif len(prefixes) == 2 and prefixes[1] in keys and event == 'end_array':
                index = keys.index(prefixes[1])
                keys_data[index] = tmp
                tmp = None

            if all(keys_data):
                all_feature_list = [
                    [
                        tf.train.Feature(bytes_list=tf.train.BytesList(value=keys_data[i][j]))
                        for j in range(len(keys_data[i]))
                    ]
                    for i in range(len(keys))
                ]
                feature_list = {
                    keys[i]: tf.train.FeatureList(feature=all_feature_list[i])
                    for i in range(len(keys))
                }
                example = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list=feature_list))
                writer.write(example.SerializeToString())
                # 读下一个数据
                keys_data = [None] * len(keys)


def build_tfrecord():
    if not os.path.exists(path):
        print('path not found')
        exit(0)
    if not os.path.exists(os.path.join(path, 'train.json')):
        print('path not found')
        exit(0)
    _, _, files = next(os.walk(path))
    for file in files:
        if not file.endswith('.json'):
            continue
        name = file.split('.')[0]
        write_to_file(os.path.join(path, file), os.path.join(path, name+'.tfrecords'))


if __name__ == '__main__':
    build_tfrecord()
