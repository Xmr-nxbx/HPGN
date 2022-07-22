import os

from tokenizers import pre_tokenizers, Tokenizer, Regex
from tokenizers.pre_tokenizers import Metaspace, Split
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from multiprocessing.dummy import Pool as ThreadPool
module_path = os.path.dirname(__file__)
vocab_size = 4000


def tokenize(code_str):
    str_list = code_str.strip().split('\n')
    re = Regex('\W')
    pre_tokenizer = pre_tokenizers.Sequence([Metaspace(), Split(re, 'isolated')])
    if os.path.exists(os.path.join(module_path, '../tokenizer.json')):
        tokenizer = Tokenizer.from_file(os.path.join(module_path, '../tokenizer.json'))
    else:
        tokenizer = None

    def get_token_inline(code_str_inline):
        if tokenizer is None:
            output = pre_tokenizer.pre_tokenize_str(code_str_inline.strip())
            return [w[0] for w in output]
            # return ' '.join([w[0] for w in output])
        else:
            output = tokenizer.encode(code_str_inline.strip())
            # restore <unk> flag (for pointer training)
            token_list = output.tokens
            unk_places = [i for i in range(len(token_list)) if token_list[i] == '<unk>']
            unk_origin_place = [output.offsets[each] for each in unk_places]
            token_list = [
                code_str_inline[unk_origin_place[unk_places.index(i)][0]: unk_origin_place[unk_places.index(i)][1]]
                if i in unk_places
                else token_list[i]
                for i in range(len(token_list))
            ]
            return token_list

    pool = ThreadPool(16)
    code_list = pool.map(get_token_inline, str_list)
    pool.close()
    pool.join()
    return code_list


def build_my_tokenizer():
    keys = ['cs', 'java']
    import json
    train_path = os.path.join(module_path, r'../dataset/train.json')
    code_list = []
    with open(train_path, 'r', encoding='utf-8') as f:
        json_data = json.loads(f.read())
        for each in json_data:
            code_list.append(each[keys[0]].strip())
            code_list.append(each[keys[1]].strip())

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    special_tokens = ['<pad>', '<unk>', '<cs>', '</cs>', '<CS>', '</CS>', '<java>', '</java>', '<JAVA>', '</JAVA>', '<cl>']
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, continuing_subword_prefix='$$')
    re = Regex('\W')
    # pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Split(re, 'isolated')])
    # tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.train_from_iterator(code_list, trainer)
    tokenizer.save(os.path.join(module_path, '../tokenizer.json'))


if __name__ == '__main__':
    # build_my_tokenizer()
    # code_str = "public static Analyzer CreateAnalyzer(换成中文 className) {\n    ￥Type clazz = Type.GetType(className);\n    try {\n        return (Analyzer)Activator.CreateInstance(clazz,\n#pragma warning disable 612, 618\nLuceneVersion.LUCENE_CURRENT);\n#pragma warning restore 612, 618\n    }\n    catch (MissingMethodException ) {\n        return (Analyzer)Activator.CreateInstance(clazz);\n    }\n}\n".strip()
    code_str = "addAll"
    output = tokenize(code_str)
    format_output = '\n'.join([''.join(each) for each in output])
    print(format_output)
    print('\n')
    print(''.join(format_output.split('$$')))
