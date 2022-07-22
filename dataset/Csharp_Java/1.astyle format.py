import os
import json
from multiprocessing.dummy import Pool as ThreadPool

keyword = ['train', 'valid', 'test']
path_to_dataset = r'D:\XuMingRui\code-to-code\translation'
path_output = './dataset'


# 数据集有bug，就得一行一行代码读进去
# need astyle environments
def astyle_outputs1(in_path, out_path):
    if not os.path.exists(in_path):
        print('path not found')
        exit(0)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        _, _, files = next(os.walk(in_path))

        command = 'astyle --stdin=%s --stdout=%s'

        for file in files:
            os.system(command % (os.path.join(in_path, file), os.path.join(path_output, file)))

    if not os.path.exists(os.path.join(out_path, keyword[0]+'.json')):
        _, _, files = next(os.walk(out_path))

        def split_data(data_str):
            split_str = '\n}\n'
            split_list = []
            split_index = 0
            while True:
                next_index = data_str.find(split_str, split_index) + len(split_str)
                if next_index - len(split_str) == -1:
                    break
                split_list.append(data_str[split_index:next_index])
                split_index = next_index
            return split_list

        for k in keyword:
            selected_files = [os.path.join(out_path, n) for n in files if k in n]
            # cs, java
            k1, k2 = selected_files[0].split('.')[-1], selected_files[1].split('.')[-1]
            with open(selected_files[0], 'r+', encoding='utf-8') as f:
                k1_str = f.read()
            with open(selected_files[1], 'r+', encoding='utf-8') as f:
                k2_str = f.read()
            k1_data_list = split_data(k1_str)
            k2_data_list = split_data(k2_str)
            if len(k1_data_list) != len(k2_data_list):
                print(len(k1_data_list), len(k2_data_list))
            dataset = [{k1: k1_data, k2: k2_data} for k1_data, k2_data in zip(k1_data_list, k2_data_list)]

            with open(os.path.join(out_path, k+'.json'), 'w', encoding='utf-8') as f:
                f.write(json.dumps(dataset))


def astyle_outputs2(in_path, out_path):
    if not os.path.exists(in_path):
        print('path not found')
        exit(0)

    tmp_path = './tmp'

    if not os.path.exists(os.path.join(out_path, keyword[0]+'.json')):

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        _, _, files = next(os.walk(in_path))

        def split_data(file_path, extension):
            with open(file_path, 'r', encoding='utf-8') as f:
                data_str_list = f.readlines()

                # 读写每一条数据
                def map_fuc(data):
                    each_data = data[0]
                    temp_file_path = data[1]
                    with open(temp_file_path, 'w', encoding='utf-8') as f1:
                        f1.write(each_data.strip()+'\n')
                    os.system('astyle %s' % temp_file_path)
                    with open(temp_file_path, 'r', encoding='utf-8') as f1:
                        return f1.read()

                tmp_file_path = [os.path.join(tmp_path, 'test%d.%s' % (i, extension)) for i in range(len(data_str_list))]
                pool = ThreadPool(20)
                data_str_list = pool.map(map_fuc, zip(data_str_list, tmp_file_path))
                pool.close()
                pool.join()

            return data_str_list

        for k in keyword:
            selected_files = [os.path.join(in_path, f) for f in files if k in f]
            file1, file2 = selected_files[0], selected_files[1]
            file1_ext, file2_ext = file1.split('.')[-1], file2.split('.')[-1]\

            data1, data2 = split_data(file1, file1_ext), split_data(file2, file2_ext)
            dataset = [{file1_ext: d1, file2_ext: d2} for d1, d2 in zip(data1, data2)]

            with open(os.path.join(out_path, k+'.json'), 'w', encoding='utf-8') as f:
                f.write(json.dumps(dataset))
    # if os.path.exists(tmp_path):
    #     os.removedirs(tmp_path)


if __name__ == '__main__':
    astyle_outputs2(path_to_dataset, path_output)
