import os
import json
import collections
from tokenizers import Tokenizer
from multiprocessing.dummy import Pool as ThreadPool

keys = ['cs_t', 'java_t']
path = './dataset'

vocab_size = 4000


def build_vocab():
    if not os.path.exists(path):
        print('path not found')
        exit(0)
    if os.path.exists(os.path.join(path, 'vocab')):
        print('vocab exists')
        exit(0)
    if not os.path.exists(os.path.join(path, 'train.json')):
        print('path not found')
        exit(0)
    if os.path.exists('./tokenizer.json'):
        tokenizer = Tokenizer.from_file(r'./tokenizer.json')
        token2id = tokenizer.get_vocab()
        tmp = {v: k for k, v in token2id.items()}
        vocab_dict = {i: tmp[i] for i in range(len(tmp))}
    else:
        tokens = []
        pool = ThreadPool(10)
        with open(os.path.join(path, 'train.json'), 'r', encoding='utf-8') as f:
            dataset = json.loads(f.read())

            def extend_data_to_tokens(index):
                tmp = []
                for k in keys:
                    for each in dataset[index][k]:
                        tmp.extend(each)
                        # tmp.extend(each.split(' '))
                tokens.extend(tmp)
            pool.map(extend_data_to_tokens, list(range(len(dataset))))
        pool.close()
        pool.join()

        vocab_dict = {0: '<pad>', 1: '<unk>',
                      2: '<cs>', 3: '</cs>', 4: '<CS>', 5: '</CS>',
                      6: '<java>', 7: '</java>', 8: '<JAVA>', 9: '</JAVA>',
                      10: '<cl>'
                      }
        result = collections.Counter(tokens).most_common(vocab_size - len(vocab_dict))
        for t, _ in result:
            vocab_dict[len(vocab_dict)] = t
    with open(os.path.join(path, 'vocab'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(vocab_dict.values())))


if __name__ == '__main__':
    build_vocab()
