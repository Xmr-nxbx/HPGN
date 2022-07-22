import tensorflow as tf
import os

keys = ['cs_t', 'java_t']
unk = '<unk>'
pad = '<pad>'
cross_line = '<cl>'
key1_flag = ['<cs>', '</cs>', '<CS>', '</CS>']
key2_flag = ['<java>', '</java>', '<JAVA>', '</JAVA>']
key1_first_word_line = ['<cs>', ['<cs>', '<CS>', '</cs>']]
key2_first_word_line = ['<java>', ['<java>', '<JAVA>', '</java>']]
module_path = os.path.dirname(__file__)


# This file summarizes all the data and uses 'get_datasets' as a callback function
def get_datasets():
    with open(os.path.join(module_path, './dataset/vocab'), 'r', encoding='utf-8') as f:
        vocab = f.read().split('\n')
    vocab = tf.convert_to_tensor(vocab)

    dataset_path = ['./dataset/train.tfrecords', './dataset/valid.tfrecords', './dataset/test.tfrecords']
    dataset_path = [os.path.join(module_path, p) for p in dataset_path]
    feature_description = {
        keys[0]: tf.io.VarLenFeature(tf.string),
        keys[1]: tf.io.VarLenFeature(tf.string),
    }

    def _parse_data(serialize_string):
        content_feature, sequence_feature = tf.io.parse_single_sequence_example(serialize_string, sequence_features=feature_description)
        # sparse tensor
        k1_data, k2_data = sequence_feature[keys[0]], sequence_feature[keys[1]]
        # k1_data, k2_data = tf.sparse.to_dense(k1_data, b''), tf.sparse.to_dense(k2_data, b'')
        return k1_data, k2_data

    raw_dataset = [tf.data.TFRecordDataset(p).map(_parse_data) for p in dataset_path]
    pad_unk_cl = [pad, unk, cross_line]
    return raw_dataset, vocab, pad_unk_cl, keys, key1_flag, key2_flag, key1_first_word_line, key2_first_word_line


def get_format_code(code_list):
    code = []
    for line in code_list:
        break_line_flag = 0
        tmp = []
        for word in line:
            break_word_flag = 0

            word = bytes.decode(word)
            if word in key1_flag or word in key2_flag:
                # end line
                if word == key1_flag[1] or word == key2_flag[1]:
                    break_word_flag = 1
                # end read
                elif word == key1_flag[3] or word == key2_flag[3]:
                    break_word_flag = 1
                    break_line_flag = 1
            elif word == cross_line:
                if len(code) != 0:
                    tmp.extend(code.pop(-1))
            elif word == pad:
                continue
            else:
                tmp.append(word)

            if break_word_flag == 1:
                break

        code.append(tmp)

        if break_line_flag == 1:
            break

    code = [''.join(line) for line in code if len(line) != 0]
    if os.path.exists(os.path.join(module_path, r'./tokenizer.json')):
        code = [''.join(line.replace("$$", "").replace('▁', " ").strip()) for line in code]  # subword prefix
    return code


def tokenizes_code_sequences(code):
    from dataset.Csharp_Java.Tokenizers import MyTokenizer
    code_list = MyTokenizer.tokenize(code)
    return code_list


if __name__ == '__main__':
    outputs = get_datasets()
    word_limit = 10
    line_limit = 60
    start = '<s>'
    end = '</s>'

    def transform(k1_data, k2_data):
        def func(data):
            print(data)
            data = data.numpy().tolist()
            data = [[word for word in line if len(word) != 0] for line in data]
            tmp = []
            for line in data:
                if len(line)+1 <= word_limit:
                    tmp.append([start] + line + [end])
                else:
                    line = [start] + line + [end]
                    # 左：计算多出单词还有多少个  右：每行单词 - start - cross_line
                    cross_line_num = (len(line) + 1 - word_limit) % (word_limit - 2)
                    tmp.append(start + line[:word_limit-1])
                    tmp.extend([
                        [start, cross_line] +
                        line[word_limit-1 + i*(word_limit-2): word_limit-1 + (i+1)*(word_limit-2)]
                        for i in range(cross_line_num)
                    ])
            if len(tmp) < line_limit:
                tmp.extend([[]] * (line_limit - len(tmp)))
            else:
                tmp = tmp[:line_limit]
            return tf.keras.preprocessing.sequence.pad_sequences(tmp, word_limit, object, 'post', 'post', pad)
        return func(k1_data), func(k2_data)

    it = outputs[0][0]
    # it = it.map(transform)
    cs, java = next(iter(it))
    cs, java = cs.numpy().tolist(), java.numpy().tolist()
    print(get_format_code(cs))
    print()
    print(get_format_code(java))
    print(java)
