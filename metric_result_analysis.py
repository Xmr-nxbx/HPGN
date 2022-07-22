from evaluator import bleu
from evaluator.CodeBLEU import get_code_bleu_scores
import importlib.util

name = {
    "Resnet-HPGN": '残差-层次注意力模型',
    "Gate-HPGN": "门控-层次注意力模型",
    "Base-HPGN": "基础-层次注意力模型",
    "Pointer-Generator": "指针生成网络"
}


def analysis():
    cs_reference_file = "dataset/Csharp_Java/cs_t_references.txt"
    java_to_cs_files = {
        "Resnet-HPGN": "dataset/Csharp_Java/model/hie_pointer_Scov_Wcov_rev_feedres_1_1_10_64/predictions.txt",
        "Gate-HPGN": "dataset/Csharp_Java/model/hie_pointer_Scov_Wcov_rev_feedgate_1_1_10_64/predictions.txt",
        "Base-HPGN": "dataset/Csharp_Java/model/hie_pointer_Scov_Wcov_rev_1_1_10_64/predictions.txt",
        "Pointer-Generator": "dataset/Csharp_Java/model/pgn_pointer_cov_rev_1_64_4176/predictions.txt"
    }

    java_reference_file = "dataset/Csharp_Java/java_t_references.txt"
    cs_to_java_files = {
        "Resnet-HPGN": "dataset/Csharp_Java/model/hie_pointer_Scov_Wcov_feedres_1_1_10_64/predictions.txt",
        "Gate-HPGN": "dataset/Csharp_Java/model/hie_pointer_Scov_Wcov_feedgate_1_1_10_64/predictions.txt",
        "Base-HPGN": "dataset/Csharp_Java/model/hie_pointer_Scov_Wcov_1_1_10_64/predictions.txt",
        "Pointer-Generator": "dataset/Csharp_Java/model/pgn_pointer_cov_1_64_4176/predictions.txt"
    }
    # show_total_result(cs_reference_file, java_to_cs_files, "c_sharp", "java to c#")
    # show_total_result(java_reference_file, cs_to_java_files, "java", "c# to java")

    show_result_of_different_length(java_reference_file, cs_reference_file, java_to_cs_files, "c_sharp", "java to c#")
    show_result_of_different_length(cs_reference_file, java_reference_file, cs_to_java_files, "java", "c# to java")


def show_total_result(ref_path, pre_path_dict, lang, title):
    ref = None
    with open(ref_path, "r+", encoding="utf-8") as f:
        ref = f.read().strip()

    bleu_result_dict = {}
    em_result_dict = {}
    codebleu_result_dict = {}

    for key in pre_path_dict.keys():
        pre = None
        with open(pre_path_dict.get(key), "r+", encoding="utf-8") as f:
            pre = f.read().strip()

        refs = [x.strip() for x in ref.split('\n')]
        pres = [x.strip() for x in pre.split('\n')]
        count = 0
        assert len(refs) == len(pres)
        for i in range(len(refs)):
            r = refs[i]
            p = pres[i]
            if r == p:
                count += 1
        em = round(count / len(refs) * 100, 2)

        bleu_result_dict[key] = round(bleu._bleu(ref_path, pre_path_dict.get(key)), 2)
        em_result_dict[key] = em
        codebleu_result_dict[key] = round(get_code_bleu_scores([ref], pre, lang) * 100, 2)

    print("=" * 10 + title + "=" * 10)
    print("model\t\t\t\t\tbleu\t\tem\t\tcodebleu")
    for key in pre_path_dict.keys():
        print(key + "\t\t" +
              str(bleu_result_dict.get(key)) + "\t\t" +
              str(em_result_dict.get(key)) + "\t\t" +
              str(codebleu_result_dict.get(key)))
        print()


spec = importlib.util.spec_from_file_location("OtherFunc",
                                              "dataset/Csharp_Java/dataset_preparation.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)


def tokenize(code):
    return foo.tokenizes_code_sequences(code)[0]


def show_result_of_different_length(origin_path, ref_path, pre_path_dict, lang, title):
    limit = 210
    step = 20
    input_len = []
    max_input_length = 0
    with open(origin_path, "r+", encoding="utf-8") as f:
        for code in f.readlines():
            code = tokenize(code.strip())
            max_input_length = max(max_input_length, len(code))
            input_len.append(len(code))
    max_input_length = min(max_input_length, limit)

    refs = None
    with open(ref_path, "r+", encoding="utf-8") as f:
        refs = [code.strip() for code in f.readlines()]

    bleu_input_length_dict = {}
    bleu_output_length_dict = {}
    codebleu_input_length_dict = {}
    codebleu_output_length_dict = {}

    for key in pre_path_dict.keys():
        pres = []
        max_len = 0
        output_len = []
        with open(pre_path_dict.get(key), "r+", encoding="utf-8") as f:
            for code in f.readlines():
                code = code.strip()
                current_len = len(tokenize(code))
                output_len.append(current_len)
                max_len = max(current_len, max_len)
                pres.append(code)

        assert len(refs) == len(pres)
        max_len = limit

        def get_list_by_max_length(length: int) -> list:
            return [0 for _ in range(0, length + 1, step)]

        input_bleu_score, input_codebleu_score, input_code_length_count = get_list_by_max_length(
            max_input_length), get_list_by_max_length(max_input_length), get_list_by_max_length(max_input_length)
        output_bleu_score, output_codebleu_score, output_code_length_count = get_list_by_max_length(
            max_len), get_list_by_max_length(max_len), get_list_by_max_length(max_len)

        def get_index_by_code_length(length: int) -> int:
            return min(int(length / step), int(limit / step))

        for i in range(len(refs)):
            input_index = get_index_by_code_length(input_len[i])
            output_index = get_index_by_code_length(output_len[i])
            cur_bleu = bleu.compute_bleu([[refs[i].split()]], [pres[i].split()], smooth=True)[0] * 100
            cur_codebleu = get_code_bleu_scores([refs[i]], pres[i], lang) * 100

            input_bleu_score[input_index] += cur_bleu
            input_codebleu_score[input_index] += cur_codebleu
            input_code_length_count[input_index] += 1

            output_bleu_score[output_index] += cur_bleu
            output_codebleu_score[output_index] += cur_codebleu
            output_code_length_count[output_index] += 1

        for i in range(len(input_code_length_count)):
            if input_code_length_count[i] != 0:
                input_bleu_score[i] /= input_code_length_count[i]
                input_codebleu_score[i] /= input_code_length_count[i]
        for i in range(len(output_code_length_count)):
            if output_code_length_count[i] != 0:
                output_bleu_score[i] /= output_code_length_count[i]
                output_codebleu_score[i] /= output_code_length_count[i]
        bleu_input_length_dict[key] = input_bleu_score
        bleu_output_length_dict[key] = output_bleu_score
        codebleu_input_length_dict[key] = input_codebleu_score
        codebleu_output_length_dict[key] = output_codebleu_score

    def draw_plot(x_label: str, y_label: str, data_dict: dict):
        import matplotlib
        import matplotlib.pyplot as plt
        import os
        from matplotlib.pyplot import MultipleLocator

        config = {
            "font.family": 'serif',
            "font.size": 20,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
            'axes.unicode_minus': False  # 处理负号，即-号
        }
        matplotlib.rcParams.update(config)
        plt.figure(figsize=(10, 6))

        ax = plt.subplot(111)  # 设置刻度字体大小

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)  # 设置坐标标签字体大小

        ax.set_xlabel(..., fontsize=16)
        ax.set_ylabel(..., fontsize=16)  # 设置轴标题字体大小
        plt.rcParams.update({'font.size': 14})  # 设置图例字体大小

        for key in data_dict.keys():
            x = [i * step for i in range(1, len(data_dict.get(key)))]
            x.append(">" + str(x[-1]))
            plt.plot(x, data_dict.get(key), marker=".", ms=5, label=name[key])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.title(title)
        # x_major_locator = MultipleLocator(2)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)

        plt.legend(loc="upper right")
        # if not os.path.exists("./figs"):
        #     os.mkdir("./figs")
        # plt.savefig("./figs/%s-%s.eps" % (title, x_label), bbox_inches='tight')
        plt.show()

    draw_plot("Input Length", "BLEU Score(%)", bleu_input_length_dict)
    draw_plot("Input Length", "CodeBLEU Score(%)", codebleu_input_length_dict)
    draw_plot("Output Length", "BLEU Score(%)", bleu_output_length_dict)
    draw_plot("Output Length", "CodeBLEU Score(%)", codebleu_output_length_dict)

    # def show_data(x_label, y_label, data_dict):
    #     print(title)
    #     x_plot = None
    #     for key in data_dict.keys():
    #         if x_plot == None:
    #             x_plot = [str(i * step) for i in range(1, len(data_dict.get(key)))]
    #             x_plot.append(">" + str(x_plot[-1]))
    #             print(' '.join(x_plot))
    #         print(name[key],x_label)
    #         print(' '.join([str(each) for each in data_dict.get(key)]))
    #     print()
    # show_data("输入长度", "CodeBLEU分数(%)", codebleu_input_length_dict)
    # show_data("输出长度", "CodeBLEU分数(%)", codebleu_output_length_dict)


if __name__ == '__main__':
    analysis()
