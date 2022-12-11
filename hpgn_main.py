import os

import numpy as np
import tensorflow as tf
import time
import importlib.util
from Network.Hierarchicalseq2seq import *

gpus = tf.config.list_physical_devices("GPU")
print(gpus)
if gpus:
    gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
    tf.config.set_visible_devices(gpu0, "GPU")
    tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用


config = {
    'vocab_size': 60000,  # not truncate again
    'hidden': 64,
    'sentence_layer': 1,
    'word_layer': 1,
    'top-k': 10,
    'pointer': True,
    'sentence_coverage': True,
    'word_coverage': True,
    'state_feed': True,
    'state_feed_mode': 0,  # 0: gate, 1: res
    'reverse_key': False,
    'lower_case': False,  # deprecated
    'force_copy': False,  # deprecated

    'epochs': 20,
    'batch_size': 40,
    'valid_step': 30,
    'input_line': 35,
    'input_word': 20,
    'output_line': 35,
    'output_word': 20,

    'datasets_dir': r'./dataset',
    'datasets_folder': None,
    'datasets_callback_file': None,
}


def select_dataset():
    if not os.path.exists(config['datasets_dir']):
        print('datasets not exist')
        exit(0)
    _, folders, _ = next(os.walk(config['datasets_dir']))
    # time.sleep(1)
    if len(folders) != 1:
        selection = input('\n'.join(['%d -> %s' % (i, folder) for i, folder in enumerate(folders)])
                          + '\nplease select data by index(other to cancel):').strip()
    else:
        selection = '0'
    if not selection.isdigit() or int(selection) not in list(range(len(folders))):
        print('exit')
        exit(0)
    config['datasets_folder'] = os.path.join(config['datasets_dir'], folders[int(selection)])

    if not os.path.exists(os.path.join(config['datasets_folder'], 'model')):
        os.makedirs(os.path.join(config['datasets_folder'], 'model'))

    # get callback function from file
    spec = importlib.util.spec_from_file_location("OtherFunc", os.path.join(config['datasets_folder'], 'dataset_preparation.py'))
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    config['datasets_callback_file'] = foo


def model_name():
    name = 'hie'
    if config['pointer']:
        name += '_pointer'
    if config['sentence_coverage']:
        name += '_Scov'
    if config['word_coverage']:
        name += '_Wcov'
        if config['force_copy']:
            name += '_Fcopy'
    if config['reverse_key']:
        name += '_rev'
    if config['state_feed']:
        name += '_feed'
        if config['state_feed_mode'] == 0:
            name += 'gate'
        else:
            name += 'res'
    name += '_%d_%d_%d_%d' % (config['sentence_layer'], config['word_layer'], config['top-k'], config['hidden'])
    return name


def entry():
    select_dataset()

    # raw_dataset: [train, valid, test] (TFRecord (k1_data, k2_data), type=tf.string)
    # vocab: tensor (tf.string)
    # pad_unk_cl: [pad, unk, cl] (str)
    # keys: [key1, key2] (str)
    # flag: [line_start, line_end, code_start, code_end]
    # first_word_line: [word, [first_line_word]]
    raw_dataset, vocab, pad_unk_cl, keys, key1_flag, key2_flag, key1_first_word_line, key2_first_word_line = \
        config['datasets_callback_file'].get_datasets()

    # vocab token preparation
    vocab = vocab[: config['vocab_size']]
    config['vocab_size'] = tf.shape(vocab)[-1]
    if config['lower_case']:
        vocab = tf.convert_to_tensor([
            bytes.decode(w).lower()
            if bytes.decode(w) not in pad_unk_cl + key1_flag + key2_flag
            else bytes.decode(w)
            for w in vocab.numpy().tolist()
        ])
    pad_id, unk_id = vocab.numpy().tolist().index(str.encode(pad_unk_cl[0])), vocab.numpy().tolist().index(str.encode(pad_unk_cl[1]))
    if config['reverse_key']:
        keys.reverse()
        key1_flag, key2_flag = key2_flag, key1_flag
        key1_first_word_line, key2_first_word_line = key2_first_word_line, key1_first_word_line
    y_first_word_id = tf.where(vocab == key2_first_word_line[0])[0, 0].numpy().tolist()
    y_first_line_id = [tf.where(vocab == each)[0, 0].numpy().tolist() for each in key2_first_word_line[1]]
    y_first_line_id += [pad_id] * (config['output_word'] - len(y_first_line_id))

    # dataset preparation
    def reverse(k1, k2):
        if config['reverse_key']:
            return k2, k1
        else:
            return k1, k2
    raw_dataset = [each.batch(config['batch_size']).map(reverse) for each in raw_dataset]
    raw_dataset[0].shuffle(config['batch_size']*3)

    # basic model preparation
    model = HierarchicalSeq2Seq(config['vocab_size'], config['hidden'], config['sentence_layer'], config['word_layer'],
                                .1, min(config['input_line'], config['top-k']), config['pointer'], config['sentence_coverage'],
                                config['word_coverage'], config['state_feed'], config['state_feed_mode'])
    optimizer = optimizers.Adam()
    now_epoch = tf.Variable(0, dtype=tf.int32)

    # init checkpoint and variables
    model_path = os.path.join(os.path.join(config['datasets_folder'], 'model'), model_name())
    print('=====================\nThe name of the currently executing model is', model_name(), '\n=====================')
    os.makedirs(model_path) if not os.path.exists(model_path) else None
    checkpoint = tf.train.Checkpoint(epoch=now_epoch, model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, model_path, checkpoint_name=model_name(), max_to_keep=3)
    if tf.train.latest_checkpoint(model_path):
        print('restore from checkpoint')
        checkpoint.restore(tf.train.latest_checkpoint(model_path)).expect_partial()
        from_inputs = input('Now Epoch / All Epoch: %d / %d\n change All Epoch to? (not digit to cancel)'
                            % (now_epoch.numpy().tolist(), config['epochs'])).strip()
        if from_inputs.isdigit():
            # now_epoch.assign(int(from_inputs))
            config['epochs'] = int(from_inputs)

    min_loss = 1e10
    valid_iter = iter(raw_dataset[1].shuffle(config['batch_size']*3).repeat(-1))

    # record loss: [epoch]
    training_loss_list = []
    valid_loss_list = []
    if os.path.exists(os.path.join(model_path, './message.log')):  # train_loss, valid_loss
        with open(os.path.join(model_path, './message.log'), 'r', encoding='utf-8') as f:
            for each in f.readlines():
                each = each.strip().split(' ')
                training_loss_list.append(float(each[0]))
                valid_loss_list.append(float(each[1]))
        training_loss_list = training_loss_list[:now_epoch.numpy().tolist()]
        valid_loss_list = valid_loss_list[:now_epoch.numpy().tolist()]
        min_loss = min(valid_loss_list)

    # training step
    for epoch in range(now_epoch.numpy().tolist(), config['epochs']):
        epoch_start = time.time()

        # train
        train_loss = []
        for step, iter_data in enumerate(raw_dataset[0]):
            step_start = time.time()
            token_id_list, word_mask_list, line_mask_list, concat_vocab = \
                hie_before_send_data_process(iter_data, [key1_flag, key2_flag], pad_unk_cl, [config['input_line'], config['output_line']], [config['input_word'], config['output_word']], vocab, config['pointer'], config['lower_case'])
            loss, out_sentence = \
                hie_auto_regressive_process(model, optimizer, token_id_list[0], token_id_list[1], word_mask_list[0], word_mask_list[1], line_mask_list[0], line_mask_list[1], y_first_line_id, y_first_word_id, unk_id, config['force_copy'], True)

            train_loss.append(loss.numpy().tolist())
            print('epoch: %d\tstep: %d\tloss:%.4f\ttime spent:%.2fs' % (epoch, step, loss.numpy().tolist(), time.time() - step_start))

            if step % 20 == 0:
                # [b, line, seq]
                out_sentence = get_token_from_vocab(out_sentence, concat_vocab)
                print('epoch: %d\tstep: %d' % (epoch, step))
                for i in range(min(3, config['batch_size'])):
                    code_line = config['datasets_callback_file'].get_format_code(out_sentence[i].numpy().tolist())
                    print('\n'.join(code_line))
                    print('-----')
        training_loss_list.append(sum(train_loss) / len(train_loss))

        # valid
        valid_loss = 0
        valid_step = config['valid_step']
        print('valid process...')
        for i in range(valid_step):
            iter_data = next(valid_iter)
            token_id_list, word_mask_list, line_mask_list, concat_vocab = \
                hie_before_send_data_process(iter_data, [key1_flag, key2_flag], pad_unk_cl, [config['input_line'], config['output_line']], [config['input_word'], config['output_word']], vocab, config['pointer'], config['lower_case'])
            loss, out_sentence = \
                hie_auto_regressive_process(model, optimizer, token_id_list[0], token_id_list[1], word_mask_list[0], word_mask_list[1], line_mask_list[0], line_mask_list[1], y_first_line_id, y_first_word_id, unk_id, config['force_copy'], False)

            valid_loss += loss.numpy().tolist()
        valid_loss = valid_loss/valid_step
        valid_loss_list.append(valid_loss)
        print('epoch: %d\tvalid_loss: %.4f\t total time: %.2fs' % (epoch, valid_loss, time.time()-epoch_start))

        # write log
        with open(os.path.join(model_path, './message.log'), 'w', encoding='utf-8') as f:
            for i in range(len(training_loss_list)):
                f.write('%f %f\n' % (training_loss_list[i], valid_loss_list[i]))

        # checkpoint config
        now_epoch.assign(epoch+1)
        if min_loss >= valid_loss or epoch == config['epochs']-1:
            min_loss = valid_loss
            manager.save(checkpoint_number=epoch+1)

    # select checkpoint model to restore
    _, _, files = next(os.walk(model_path))
    files = ['.'.join(f.split('.')[:-1]) for f in files if "index" in f]
    if len(files) == 0:
        print('no checkpoint exists! using latest model')
    else:
        print('\n'.join(["%d -> %s" % (i, file) for i, file in enumerate(files)]))
        if len(valid_loss_list) != 0:
            recommend_ckpt = valid_loss_list.index(min(valid_loss_list)) + 1
            selection = input("(recommend ckpt-%d)\nselect index of ckpt to generate output:" % recommend_ckpt)
        else:
            selection = input("select index of ckpt to generate output:")
        if selection.isdigit():
            selection = int(selection)
            if selection < len(files):  # index
                file = files[selection]
            else:  # epoch
                file = [f for f in files if f.endswith(str(selection))][0]
            checkpoint.restore(os.path.join(model_path, file)).expect_partial()
        else:
            print('not select. using latest model')

    # generation step (not finish)
    print('\n generate process')
    for step, iter_data in enumerate(raw_dataset[2]):
        step_start = time.time()
        token_id_list, word_mask_list, line_mask_list, concat_vocab = \
            hie_before_send_data_process(iter_data, [key1_flag, key2_flag], pad_unk_cl,
                                         [config['input_line'], config['output_line']],
                                         [config['input_word'], config['output_word']], vocab, config['pointer'],
                                         config['lower_case'])
        loss, out_sentence = \
            hie_auto_regressive_process(model, optimizer, token_id_list[0], token_id_list[1], word_mask_list[0], word_mask_list[1],
                                        line_mask_list[0], line_mask_list[1], y_first_line_id, y_first_word_id,
                                        unk_id, config['force_copy'], False)

        print('step: %d\tloss: %.4f\ttime spent:%.2fs' % (step, loss.numpy().tolist(), time.time() - step_start))
        out_sentence = get_token_from_vocab(out_sentence, concat_vocab).numpy().tolist()

        for i in range(config['batch_size']):
            out_sentence[i] = ' '.join(config['datasets_callback_file'].get_format_code(out_sentence[i]))
        predictions = '\n'.join(out_sentence) + '\n'

        with open(os.path.join(model_path, './predictions.txt'), 'a', encoding='utf-8') as f2:
            f2.write(predictions)


if __name__ == '__main__':
    entry()
