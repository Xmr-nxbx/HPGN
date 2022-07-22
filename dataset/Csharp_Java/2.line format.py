import json
import os
from multiprocessing.dummy import Pool as ThreadPool
from Tokenizers import MyTokenizer as Tokenizer

keys = ['cs', 'java']
path = './dataset'


# before tokenize code, you need to manually format C# preprocessor directives
# like ...#endif ... => ...\n #endif \n ...
def line_format():
    if not os.path.exists(path):
        print('path not found')
        exit(0)
    _, _, files = next(os.walk(path))
    if len(files) == 0:
        print('path not found')
        exit(0)

    for each in files:
        if not each.endswith('.json'):
            continue
        with open(os.path.join(path, each), 'r', encoding='utf-8') as f:
            dataset = json.loads(f.read())

        # warning: run 'MyTokenizer.py' first
        def func(i):
            dataset[i]['%s_t' % keys[0]] = Tokenizer.tokenize(dataset[i][keys[0]])
            dataset[i]['%s_t' % keys[1]] = Tokenizer.tokenize(dataset[i][keys[1]])
            if i % 10 == 0:
                print(each, i)
        pool = ThreadPool(10)
        pool.map(func, list(range(len(dataset))))
        pool.close()
        pool.join()

        with open(os.path.join(path, each), 'w', encoding='utf-8') as f:
            f.write(json.dumps(dataset))


if __name__ == '__main__':
    line_format()
