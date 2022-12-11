import numpy as np
import tensorflow as tf
from tensorflow.keras import *

from Network.GRUseq2seq import *


class StatesFeedMechanism(layers.Layer):
    def __init__(self, sentence_layer_num, hidden):
        super(StatesFeedMechanism, self).__init__()
        self.sentence_layer_num = sentence_layer_num
        self.hidden = hidden

        self.W1 = layers.Dense(hidden, use_bias=False)
        self.W2 = layers.Dense(hidden, use_bias=False)
        self.W3 = [layers.Dense(hidden) for _ in range(sentence_layer_num)]
        self.V1 = [layers.Dense(hidden, activation=tf.nn.tanh) for _ in range(sentence_layer_num)]

    def call(self, encode_word_states, decode_word_states, sentence_states, *args, **kwargs):
        """
        :param encode_word_states: [b, hid]
        :param decode_word_states: [b, hid]
        :param sentence_states: [sentence_layer_num, b, hid]
        :return: [sentence_layer_num, b, hid]
        """
        x = self.W1(encode_word_states) + self.W2(decode_word_states)  # [b, hid]

        states = tf.TensorArray(dtype=tf.float32, size=self.sentence_layer_num)
        for i in range(self.sentence_layer_num):
            weight = tf.nn.sigmoid(x + self.W3[i](sentence_states[i]))  # [b, hid]
            state_i = (1.-weight) * sentence_states[i] + weight * self.V1[i](decode_word_states)  # [b, hid]
            states = states.write(i, state_i)

        return states.stack()


class HierarchicalSeq2Seq(Model):
    def __init__(self, vocab_size, hidden, sentence_layer_num=1, word_layer_num=1, dropout=0.1, sentence_top_k=-1,
                 pointer=True, sentence_coverage=True, word_coverage=True, states_feed=True, states_feed_mode=0, pretrain_model=None):
        super(HierarchicalSeq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.sentence_layer_num = sentence_layer_num
        self.word_layer_num = word_layer_num
        self.sentence_top_k = sentence_top_k
        self.pointer = pointer
        self.word_coverage = word_coverage
        self.sentence_coverage = sentence_coverage
        self.states_feed = states_feed
        self.states_feed_mode = states_feed_mode

        self.sentence_encoder = GRUBiEncoder(word_layer_num * hidden, sentence_layer_num, dropout)
        self.sentence_decoder = GRUDecoder(word_layer_num * hidden, sentence_layer_num, dropout, sentence_coverage)
        self.reduce_sentence_weight = [layers.Dense(word_layer_num * hidden, activation='relu') for _ in range(sentence_layer_num)]

        if pretrain_model is None:
            self.embedding = layers.Embedding(vocab_size, hidden)
            self.word_encoder = GRUBiEncoder(hidden, word_layer_num, dropout)
            self.word_decoder = GRUDecoder(hidden, word_layer_num, dropout, word_coverage)
            self.reduce_word_weight = [layers.Dense(hidden, activation='relu') for _ in range(word_layer_num)]
            self.fc = layers.Dense(vocab_size, activation=tf.nn.softmax, use_bias=False)
            if pointer:
                self.pointer_generator = PointerGenerator()
        else:
            self.embedding = pretrain_model.embedding
            self.word_encoder = pretrain_model.encoder
            self.word_decoder = pretrain_model.decoder
            self.reduce_word_weight = pretrain_model.reduce_weight
            self.fc = pretrain_model.fc
            if pointer:
                self.pointer_generator = pretrain_model.pointer_generator

        if self.states_feed:
            if self.states_feed_mode == 0:
                self.states_feed_mechanism = StatesFeedMechanism(self.sentence_layer_num, self.word_layer_num*self.hidden)
            else:
                self.feed_forward = [layers.Dense(self.word_layer_num*self.hidden) for _ in range(self.sentence_layer_num)]

    def embed_words(self, x):
        """
        return x embedding vector

        :param x: x [b, seq]
        :return: embed_x [b, seq, hid]
        """
        return self.embedding(x)

    # step 1
    def word_encode_process(self, sentences, training=None):
        """
        return word_encoder hidden sequence and state

        :param training: bool, for dropout
        :param sentences: [b, line, seq]
        :return: word_encode_sequence [b, line, word_seq, 2*hid], word_encode_state [word_layers, b, line, hid]
        """
        batch_size = tf.shape(sentences)[0]
        line_num = tf.shape(sentences)[1]
        seq_len = tf.shape(sentences)[2]
        sentences = tf.reshape(sentences, [-1, seq_len])
        # [b * line, seq, hidden]
        encode_embed = self.embed_words(sentences)
        # [b * line, seq, 2*hidden], word_layer * [b * line, hidden] * 2
        word_encode_sequence, word_forward, word_backward = self.word_encoder(encode_embed, training=training)
        # [word_layer, b * line, hidden]
        word_states = tf.TensorArray(dtype=tf.float32, size=self.word_layer_num)
        for i in range(self.word_layer_num):
            state = tf.concat([word_forward[i], word_backward[i]], -1)
            state = self.reduce_word_weight[i](state)
            word_states = word_states.write(i, state)
        word_states = word_states.stack()

        # restore
        word_encode_sequence = tf.reshape(word_encode_sequence, [batch_size, line_num, seq_len, 2*self.hidden])
        word_states = tf.reshape(word_states, [self.word_layer_num, batch_size, line_num, self.hidden])
        # word_encode_sequence [b, line, word_seq, 2*hid]
        # word_states [word_layer, b, line, hid]
        return word_encode_sequence, word_states

    # step 2
    def sentence_encode_process(self, word_states, training=None):
        """
        return sentence_encoder hidden sequence and state

        :param word_states: [word_layer, b, line, hid]
        :param training: bool, for dropout
        :return: line_encode_sequence, line_states
        """
        batch_size = tf.shape(word_states)[1]
        line_num = tf.shape(word_states)[2]
        # [b, line, word_layer*hid]
        word_states = tf.transpose(word_states, [1, 2, 0, 3])
        word_states = tf.reshape(word_states, [batch_size, line_num, self.word_layer_num * self.hidden])
        # [b, line, word_layer*hid*2], sentence_layer * [b, word_layer*hid] * 2
        sentence_encode_sequence, sentence_forward, sentence_backward = self.sentence_encoder(word_states, training=training)
        # [sentence_layer, b, word_layer*hid]
        sentence_states = tf.TensorArray(dtype=tf.float32, size=self.sentence_layer_num)
        for i in range(self.sentence_layer_num):
            state = tf.concat([sentence_forward[i], sentence_backward[i]], -1)
            state = self.reduce_sentence_weight[i](state)
            sentence_states = sentence_states.write(i, state)
        sentence_states = sentence_states.stack()

        # sentence_encode_sequence [batch_size, line_num, word_layer_num * hidden * 2]
        # sentence_states [sentence_layer_num, batch_size, word_layer_num * hidden]

        return sentence_encode_sequence, sentence_states

    # step 3.0 state feed method
    def states_feed_process(self, encode_word_states, decode_word_states, sentence_states):
        """
        return states of the t-step sentence

        :param encode_word_states: [word_layer_num, batch_size, 1, hidden]
        :param decode_word_states: [word_layer_num, batch_size, hidden]
        :param sentence_states: [sentence_layer_num, batch_size, hidden * word_layer_num]
        :return: new sentence_states: [sentence_layer_num, batch_size, hidden * word_layer_num]
        """
        batch_size = tf.shape(decode_word_states)[1]

        # [b, hidden*word_layer]
        encode_word_states = tf.reshape(tf.transpose(encode_word_states, [1, 0, 2, 3]), [batch_size, self.word_layer_num * self.hidden])
        decode_word_states = tf.reshape(tf.transpose(decode_word_states, [1, 0, 2]), [batch_size, self.word_layer_num * self.hidden])

        if self.states_feed_mode != 0:
            # [sentence_layer, b, hidden*word_layer] for memory
            # combined_states = sentence_states + decode_word_states
            # [sentence_layer, b, hidden*word_layer*2]
            state_list = tf.TensorArray(dtype=tf.float32, size=self.sentence_layer_num)
            for i in range(self.sentence_layer_num):
                combined_states = tf.concat([sentence_states[i], encode_word_states, decode_word_states], axis=-1)
                state_list = state_list.write(i, self.feed_forward[i](combined_states))
            # [sentence_layer, b, hidden*word_layer]
            combined_states = state_list.stack()
            sentence_states = tf.nn.relu(combined_states + sentence_states)
        else:
            # compute
            sentence_states = self.states_feed_mechanism(encode_word_states, decode_word_states, sentence_states)
        return sentence_states

    # step 3.1
    def sentence_decode_process(self, previous_sentence, sentence_encode_sequence, sentence_mask, sentence_states, decode_previous_word_states=None, sentence_coverage_vector=None, training=None):
        """
        return word_states, sentence_states, sentence_context_weight, sentence_next_coverage_vector
        (word_states, sentence_context_weight) for word decoder (states, top-k select)
        (sentence_states, sentence_next_coverage_vector) for sentence decoder (states, cov-loss)

        :param previous_sentence: [b, word_out_seq]
        :param sentence_encode_sequence: [b, line, word_layer_num * hid * 2]
        :param sentence_mask: [b, line]
        :param sentence_states: [sentence_layer_num, b, word_layer_num * hid]
        :param decode_previous_word_states: [word_layer_num, b, hid]
        :param sentence_coverage_vector: [b, line, 1] or None
        :param training: bool for dropout
        :return: word_states, sentence_states, sentence_context_weight, sentence_next_coverage_vector
        """
        batch_size = tf.shape(previous_sentence)[0]
        word_out_seq = tf.shape(previous_sentence)[1]

        # step 1 : encode to get previous sentence
        previous_sentence = tf.expand_dims(previous_sentence, 1)
        # [b, 1, word_seq, hidden], [word_layer, b, 1, hidden]
        _, previous_sentence_word_states = self.word_encode_process(previous_sentence, training=training)
        previous_sentence_word_states = previous_sentence_word_states

        # step 2.0 : state feed process
        if self.states_feed:
            sentence_states = self.states_feed_process(previous_sentence_word_states, decode_previous_word_states, sentence_states)

        # step 2.1 : reshape for current input
        # [word_layer, b, 1, hidden]=>[b, 1, word_layer * hidden]
        previous_word_states = tf.transpose(previous_sentence_word_states, [1, 2, 0, 3])
        previous_word_states = tf.reshape(previous_word_states, [batch_size, 1, self.word_layer_num*self.hidden])

        # [b, 1, word_layer*hid], [sentence_layer, b, word_layer*hid], _, [b, line, 1], [b, line, 1] or None
        sentence_hidden, sentence_states, _, sentence_context_weight, sentence_next_coverage_vector = \
            self.sentence_decoder(previous_word_states, sentence_encode_sequence, sentence_mask, sentence_states, sentence_coverage_vector, training=training)
        # [b, word_layer, hidden]=>[word_layer, b, hidden]
        word_states = tf.reshape(sentence_hidden, [batch_size, self.word_layer_num, self.hidden])
        word_states = tf.transpose(word_states, [1, 0, 2])

        # word_states [word_layer_num, b, hidden]
        # sentence_states [sentence_layer_num, b, word_layer_num * hidden]
        # sentence_context_weight [b, line, 1]
        # sentence_next_coverage_vec [b, line, 1] or None
        return word_states, sentence_states, sentence_context_weight, sentence_next_coverage_vector

    # step 4 (data preparation)
    def before_word_decode_process(self, sentence_context_weight, word_encode_sequence, word_encode_mask, encode_sentences):
        """
        top-k selection, return weight, sequence, mask, encode_sentences for current output_sentence
        encode_sentences is id relocated sentences, generated from concat-vocab

        :param sentence_context_weight: [b, line, 1]
        :param word_encode_sequence: [b, line, seq, 2*hidden]
        :param word_encode_mask: [b, line, seq]
        :param encode_sentences: [b, line, seq]
        :return: (sentence_context_weight, word_encode_sequence, word_encode_mask, encode_sentences), indices
        """
        batch_size = tf.shape(sentence_context_weight)[0]
        line_num = tf.shape(sentence_context_weight)[1]
        # set appropriate line num, and select
        if self.sentence_top_k == -1:
            select_line_num = line_num
        else:
            select_line_num = tf.minimum(line_num, self.sentence_top_k)
        # sentence_context_weight [b, line]
        sentence_context_weight = tf.squeeze(sentence_context_weight, -1)
        # [b, top-k]
        sentence_context_weight, indices = tf.math.top_k(sentence_context_weight, select_line_num, False)

        # [b, b*top-k, 2]
        new_indices = tf.TensorArray(dtype=tf.int32, size=batch_size)
        for i in range(batch_size):
            tmp = tf.TensorArray(dtype=tf.int32, size=select_line_num)
            for j in range(select_line_num):
                tmp = tmp.write(j, [i, indices[i, j]])
            new_indices = new_indices.write(i, tmp.stack())
        new_indices = new_indices.stack()

        # [b, top - k, word_seq, 2 * hidden], [b, top - k, word_seq]
        word_encode_sequence = tf.gather_nd(word_encode_sequence, new_indices)
        word_encode_mask = tf.gather_nd(word_encode_mask, new_indices)
        encode_sentences = tf.gather_nd(encode_sentences, new_indices)

        return sentence_context_weight, word_encode_sequence, word_encode_mask, encode_sentences, new_indices

    # step 5 decode_process
    def word_decode_process(self, previous_word, sentence_context_weight, word_encode_sequence, word_encode_mask, word_states, encode_sentences=None, word_coverage_vector=None, training=None):
        """
        return distribution of the t-step word, and (states, context_weight, next_coverage)

        :param previous_word: [b, 1]
        :param sentence_context_weight: [b, line]
        :param word_encode_sequence: [b, line, word_seq, 2*hidden]
        :param word_encode_mask: [b, line, word_seq]
        :param word_states: [word_layer_num, b, hidden]
        :param encode_sentences: [b, line, seq]
        :param word_coverage_vector: [b, word_seq, 1]
        :param training: bool for dropout
        :return: distribution, word_states, word_context_weight, next_cover
        """
        # [b, 1, hid]
        previous_word_embed = self.embed_words(previous_word)

        # [b, 1, hid], [word_layer, b, hid], [b, 1, 2*hid], [b, line, seq, 1], [b, line, seq, 1]
        decode_out, word_states, word_context_vector, word_context_weight, word_next_coverage_vector \
            = self.word_decoder.decode_from_multi_line_process(previous_word_embed, word_encode_sequence, word_encode_mask, word_states, word_coverage_vector, sentence_context_weight, training=training)
        # distribution [b, class]
        distribution = tf.squeeze(self.fc(decode_out), -2)

        # record
        distribution_list = [distribution]

        if self.pointer:
            distribution, pointer_distribution, p_gen = self.pointer_generator_process(distribution, word_context_vector, word_states, previous_word_embed, word_context_weight, encode_sentences, word_encode_mask)
            distribution_list.append(pointer_distribution)
            distribution_list.append(p_gen)
        # distribution [b, class] or [b, class + top-k * line]
        # word_states [word_layer, b, hid]
        # word_context_weight [b, line, seq, 1]
        # word_next_coverage_vector [b, line, seq, 1]
        # distribution_list: [vocab_distribution, pointer_distribution, p_gen]
        return distribution, word_states, word_context_weight, word_next_coverage_vector, distribution_list

    def pointer_generator_process(self, distribution, word_context_vector, word_states, word_embed, attention_weight, encode_sentences, word_encode_mask):
        """
        calculate and gather the output distribution from the concat-vocab

        :param distribution: [b, class]
        :param word_context_vector: [b, 1, 2*hid]
        :param word_states: [word_layer, b, hid]
        :param word_embed: [b, 1, hidden]
        :param attention_weight: [b, line, seq, 1]
        :param encode_sentences: [b, line, seq]
        :param word_encode_mask: [b, line, seq]
        :return: concat_distribution [b, class + line * word_seq]
        """
        batch_size = tf.shape(distribution)[0]
        # [b, 2*hid]
        c_v = tf.squeeze(word_context_vector, -2)
        # [b, word_layer * hid]
        s = tf.reshape(tf.transpose(word_states, [1, 0, 2]), [batch_size, self.word_layer_num * self.hidden])
        # [b, hid]
        embed = tf.squeeze(word_embed, -2)
        # [b, 1]
        p_gen = self.pointer_generator(c_v, s, embed)
        # [b, class + line * word_seq]
        distribution, pointer_distribution = get_hie_distribution_from_both(encode_sentences, word_encode_mask, attention_weight, distribution, p_gen)
        return distribution, pointer_distribution, p_gen

    def call(self, inputs, training=None, mask=None):
        pass


def forced_copying_process(y_output_id, x_token_id, x_token_mask, x_sentence_context_weight, word_coverage_vector, unk_num=1):
    """
    forced copying word from x_token_id by coverage weight, reduce unk

    :param y_output_id: [b]
    :param x_token_id: [b, line, seq]
    :param x_token_mask: [b, line, seq]
    :param x_sentence_context_weight: [b, line]  provide sentence selection's weights for words
    :param word_coverage_vector: [b, line, seq, 1]
    :param unk_num: 1
    :return: y_output_id, loss_weight same as y_output_id
    """
    indices = tf.cast(tf.where(y_output_id == unk_num), tf.int32)  # [unk_num, 1]
    loss_weight = tf.ones(tf.shape(y_output_id), tf.float32)  # [unk_num]
    if tf.shape(indices)[0] == 0:
        return y_output_id, loss_weight

    unk_token_num = tf.shape(indices)[0]
    # [unk, line, seq] -> [unk, line*seq]
    x_sentence_context_weight = tf.broadcast_to(tf.expand_dims(x_sentence_context_weight, -1), tf.shape(x_token_id))  # [b, line, seq]
    x_sentence_context_weight = tf.reshape(tf.gather_nd(x_sentence_context_weight, indices), [unk_token_num, -1])  # [unk, line, seq] -> [unk, -1]
    word_coverage_vector = tf.reshape(tf.gather_nd(tf.squeeze(word_coverage_vector, -1), indices), [unk_token_num, -1])
    x_token_id = tf.reshape(tf.gather_nd(x_token_id, indices), [unk_token_num, -1])
    x_token_mask = tf.reshape(tf.gather_nd(x_token_mask, indices), [unk_token_num, -1])

    # [unk, line*seq]
    # step1: calculate softmax weight. The weight of mask places is 0 (before: e^-∞)
    weight = tf.nn.softmax(word_coverage_vector + (-1e9 * tf.cast(x_token_mask, word_coverage_vector.dtype)), axis=-1)
    # step2: calculate the weight of low-frequency words. Vec1 - weight
    weight = (1. - tf.cast(x_token_mask, weight.dtype)) - weight
    # step3: because ∑ Vec1 = line*seq, ∑ softmax = 1, and we should shift interval to [0, 1]
    #        so weight = weight / (line*seq - 1)
    weight = weight / tf.cast(tf.shape(weight)[-1] - 1, tf.float32)
    # step4: introduce sentence selection's weight
    weight = weight * x_sentence_context_weight

    # [unk, 1]
    random_indices = tf.random.categorical(tf.math.log(weight), 1, dtype=tf.int32)
    new_random_indices = tf.TensorArray(tf.int32, size=unk_token_num)
    for i in range(unk_token_num):
        new_random_indices = new_random_indices.write(i, [i, random_indices[i, 0]])
    # [unk, 2]
    random_indices = new_random_indices.stack()
    # [unk]
    random_selected = tf.gather_nd(x_token_id, random_indices)
    random_selected_weight = tf.gather_nd(weight, random_indices)

    # mask random_selected_weight
    # give up training : keep training = 6 : 4 => [0, .6) : [.6: 1)
    random_probability = tf.random.uniform(tf.shape(random_selected_weight), 0, 1)
    random_selected_weight = tf.where(.6 > random_probability, tf.zeros_like(random_selected_weight), random_selected_weight)

    y_output_id = tf.tensor_scatter_nd_update(y_output_id, indices, random_selected)
    loss_weight = tf.tensor_scatter_nd_update(loss_weight, indices, random_selected_weight)
    return y_output_id, loss_weight


def get_hie_distribution_from_both(source_id, source_mask, attention_weight, vocab_distribution, p_gen):
    """
    hierarchical pointer-generator distribution generate step

    :param source_id: [b, top-k, seq]
    :param source_mask: [b, top-k, seq] pad: 1
    :param attention_weight: [b, top-k, seq, 1]
    :param vocab_distribution: [b, class]
    :param p_gen: [b, 1]
    :return: [b, class + top-k*seq]
    """
    batch_size = tf.shape(source_id)[0]
    attention_weight = tf.squeeze(attention_weight, -1)


    source_id = tf.reshape(source_id, [batch_size, -1])  # [b, seq]
    source_mask = tf.reshape(source_mask, [batch_size, -1])  # [b, seq]
    attention_weight = tf.reshape(attention_weight, [batch_size, -1])  # [b, seq]

    # attention_weight * effect * (1-p_gen) => [b, seq] * [b, seq] * [b, 1] => [b, seq]
    attention_weight = attention_weight * tf.cast(1 - source_mask, tf.float32) * (1. - p_gen)

    seq_len = tf.shape(source_id)[1]
    class_num = tf.shape(vocab_distribution)[1]

    expand_class_num = class_num + seq_len

    # 1.把原始序列扩增 [b, class] => [b, expand_class]
    vocab_distribution = tf.pad(p_gen * vocab_distribution, paddings=[[0, 0], [0, seq_len]])

    # 2.计算原始序列概率上的分布
    source_id = tf.expand_dims(source_id, axis=-1)
    attention_distribution = tf.TensorArray(dtype=tf.float32, size=batch_size)
    for i in range(batch_size):
        distribution = tf.tensor_scatter_nd_add(tf.zeros([expand_class_num]), indices=source_id[i], updates=attention_weight[i])
        attention_distribution = attention_distribution.write(i, distribution)
    attention_distribution = attention_distribution.stack()
    final_distribution = vocab_distribution + attention_distribution

    return final_distribution, attention_distribution


def hie_before_send_data_process(dataset, flags, pad_unk_cl, line_limits, word_limits, vocab_token, pointer=True, lower_case=False):
    if isinstance(dataset[0], tf.SparseTensor):
        dataset = [tf.sparse.to_dense(each).numpy().tolist() for each in dataset]
        dataset = [[[[bytes.decode(w) for w in line] for line in batch] for batch in each_data] for each_data in dataset]
    batch_size = len(dataset[0])
    starts = [each[0] for each in flags]
    ends = [each[1] for each in flags]
    start_lines = [[each[0], each[2], each[1]] for each in flags]
    end_lines = [[each[0], each[3], each[1]] for each in flags]
    import math

    def convert_to_standard_input(indices):
        data_index = indices[0]
        batch_index = indices[1]
        single_batch_data = dataset[data_index][batch_index]
        single_batch_data = [
            [
                word.lower() if lower_case else word
                for word in line if len(word) != 0
             ]
            for line in single_batch_data
        ]
        single_batch_data = [line for line in single_batch_data if len(line) != 0]

        # tmp = [start_lines[data_index]]
        tmp = []
        for line in single_batch_data:
            if len(line) < word_limits[data_index]:
                tmp.append(line + [ends[data_index]])
            else:
                word_limit = word_limits[data_index]

                newList = [ line[:word_limit-1] + [ends[data_index]] ]
                line = line[word_limit-1:]
                while len(line) > 0:
                    newList += [ [pad_unk_cl[2]] + line[:word_limit-2] + [ends[data_index]] ]
                    line = line[word_limit-2:]

                tmp.extend(newList)
        tmp.append(end_lines[data_index])
        if len(tmp) < line_limits[data_index]:
            tmp.extend([[]] * (line_limits[data_index] - len(tmp)))
        else:
            tmp = tmp[:line_limits[data_index]]
        return tf.keras.preprocessing.sequence.pad_sequences(tmp, word_limits[data_index], object, 'post', 'post', pad_unk_cl[0])

    def pad_sequence(index):
        data_index = index
        max_line_num = 0
        for each in dataset[data_index]:
            max_line_num = max( min(len(each), line_limits[data_index]), max_line_num)
        for i in range(len(dataset[data_index])):
            each = dataset[data_index][i].tolist()
            if len(each) < max_line_num:
                each.extend([[pad_unk_cl[0]] * word_limits[data_index]] * (max_line_num - len(each)))
            else:
                each = each[:max_line_num]
            dataset[data_index][i] = each
        return dataset[data_index]

    pool = ThreadPool(batch_size)
    dataset = [pool.map(convert_to_standard_input, [[i, j] for j in range(len(dataset[i]))])
               for i in range(len(dataset))]
    # dataset = pool.map(pad_sequence, [i for i in range(len(dataset))])
    pool.close()
    pool.join()

    dataset = [tf.stack(each, 0) for each in dataset]

    pad_num = vocab_token.numpy().tolist().index(str.encode(pad_unk_cl[0]))
    unk_num = vocab_token.numpy().tolist().index(str.encode(pad_unk_cl[1]))

    def build_token_id(data_token):
        data_shape = tf.shape(data_token)
        data_token = tf.expand_dims(data_token, -1)
        indices = tf.cast(tf.where(data_token == vocab_token), tf.int32)
        word_indices = indices[..., :-1]
        word_updates = indices[..., -1]
        return tf.tensor_scatter_nd_update(tf.fill(data_shape, unk_num), word_indices, word_updates)
    token_id_list = [build_token_id(each) for each in dataset]
    word_mask_list = [tf.cast(each == pad_num, tf.int32) for each in token_id_list]

    def build_line_mask(data_id, length):
        mask_compute = data_id @ tf.ones([length, 1], tf.int32)
        line_mask = tf.squeeze(mask_compute, -1) == length
        return tf.cast(line_mask, tf.int32)
    line_mask_list = [build_line_mask(word_mask_list[i], word_limits[i]) for i in range(len(word_mask_list))]
    # print(dataset[0][0])
    # print(word_mask_list[0][0])
    # print(line_mask_list[0][0])
    # print(dataset[1][0])
    # print(word_mask_list[1][0])
    # print(line_mask_list[1][0])

    concat_vocab = vocab_token
    if pointer:
        extra_vocab = get_extra_vocab(dataset[0], vocab_token)
        concat_vocab = get_concat_vocab(extra_vocab, vocab_token)
        token_id_list = [get_token_id_from_concat_vocab(each, concat_vocab, unk_num) for each in dataset]
    return token_id_list, word_mask_list, line_mask_list, concat_vocab


@tf.function
def hie_auto_regressive_process(model: HierarchicalSeq2Seq, optimizer, x_token_id, y_token_id, x_word_mask, y_word_mask,
                                x_line_mask, y_line_mask, y_first_line, y_first_id, unk_num=1, force_copy=True, training=None):
    """
    training data in one batch
    return loss and outputs(need decode)

    :param model: HierarchicalSeq2Seq
    :param optimizer: tf.keras.optimizers
    :param x_token_id: [b, line, seq_in]
    :param y_token_id: [b, line, seq_out]
    :param x_word_mask: [b, line, seq_in], pad place: 1
    :param y_word_mask: [b, line, seq_out], pad place: 1
    :param x_line_mask: [b, line], all words inline: 1
    :param y_line_mask: [b, line], all words inline: 1
    :param y_first_line: [seq_out], help to generate next line
    :param y_first_id: number, help to generate next word in same line
    :param unk_num: number, unk number, for pointer generator network input
    :param force_copy: bool for y_id relocation(need for word coverage)
    :param training: bool for dropout
    :return: loss, grads, out_sentence[b, line, seq_out](Tensor.int32), concat_vocab (if pointer==True)
    """
    batch_size = tf.shape(x_token_id)[0]
    # in_line_len = length_config[0]
    # in_seq_len = length_config[1]
    # out_line_len = length_config[2]
    # out_seq_len = length_config[3]
    in_line_len = tf.shape(x_token_id)[1]
    in_seq_len = tf.shape(x_token_id)[2]
    out_line_len = tf.shape(y_token_id)[1]
    out_seq_len = tf.shape(y_token_id)[2]

    sentence_lambda = 1.
    word_lambda = 1.

    with tf.GradientTape() as tape:
        input_x_token_id = x_token_id
        if model.pointer:
            input_x_token_id = tf.where(input_x_token_id < model.vocab_size, input_x_token_id, tf.fill([batch_size, in_line_len, in_seq_len], unk_num))
        # [b, line, word_seq, 2*hid], [word_layers, b, line, hid]
        word_encode_sequence, word_states = model.word_encode_process(input_x_token_id, training=training)
        # [b, line, word_layer*hid*2], [sentence_layer, b, word_layer*hid]
        sentence_encode_sequence, sentence_states = model.sentence_encode_process(word_states, training=training)

        # list: [b,seq]
        out_sentence = tf.TensorArray(dtype=tf.int32, size=out_line_len+1)
        out_sentence = out_sentence.write(0, tf.broadcast_to(y_first_line, [batch_size, out_seq_len]))

        # coverage
        sentence_coverage_vector = tf.zeros([batch_size, in_line_len, 1])
        all_word_coverage_vector = tf.zeros([batch_size, in_line_len, in_seq_len, 1])
        sentence_coverage_penalty = tf.TensorArray(dtype=tf.float32, size=out_line_len)
        word_coverage_penalty = tf.TensorArray(dtype=tf.float32, size=out_line_len)

        # loss record
        all_distribution = tf.TensorArray(dtype=tf.float32, size=out_line_len)
        all_y_true = tf.TensorArray(dtype=tf.int32, size=out_line_len)
        all_distribution_loss_weight = tf.TensorArray(dtype=tf.float32, size=out_line_len)

        decode_previous_word_states = tf.zeros([model.word_layer_num, batch_size, model.hidden])

        for line_index in range(out_line_len):
            previous_sentence = out_sentence.read(line_index)
            # if previous has some extra vocab tokens, restore to normal tokens
            if model.pointer:
                previous_sentence = tf.where(previous_sentence < model.vocab_size, previous_sentence, tf.fill([batch_size, out_seq_len], unk_num))

            word_states, sentence_states, sentence_context_weight, sentence_next_coverage_vector = \
                model.sentence_decode_process(previous_sentence, sentence_encode_sequence, x_line_mask, sentence_states, decode_previous_word_states, sentence_coverage_vector, training=training)

            # top-k data preparation
            k_sentence_context_w, k_word_encode_seq, k_x_word_mask, k_x_token_id, k_indices = \
                model.before_word_decode_process(sentence_context_weight, word_encode_sequence, x_word_mask, x_token_id)

            # each time append [b], total (line_index - 1)
            if model.sentence_coverage:
                sentence_coverage_penalty = sentence_coverage_penalty.write(line_index, tf.reduce_sum(tf.reduce_min(tf.concat([sentence_coverage_vector, sentence_context_weight], -1), -1), -1))
                sentence_coverage_vector = sentence_next_coverage_vector

            # also gather top-k word_coverage_vector
            word_coverage_vector = tf.gather_nd(all_word_coverage_vector, k_indices)  # [b,top-k,seq_len,1]
            out_word = tf.TensorArray(dtype=tf.int32, size=out_seq_len+1)
            out_word = out_word.write(0, tf.fill([batch_size], y_first_id))
            word_penalty = tf.TensorArray(dtype=tf.float32, size=out_seq_len)

            # word loss record
            word_distribution = tf.TensorArray(dtype=tf.float32, size=out_seq_len)
            word_y_true = tf.TensorArray(dtype=tf.int32, size=out_seq_len)
            word_distribution_loss_weight = tf.TensorArray(dtype=tf.float32, size=out_seq_len)

            for word_index in range(out_seq_len):
                previous_word = out_word.read(word_index)
                # restore to normal token
                if model.pointer:
                    previous_word = tf.where(previous_word < model.vocab_size, previous_word, tf.fill([batch_size], unk_num))
                previous_word = tf.expand_dims(previous_word, -1)

                distribution, word_states, word_context_weight, next_cover, _ = \
                    model.word_decode_process(previous_word, k_sentence_context_w, k_word_encode_seq, k_x_word_mask, word_states, k_x_token_id, word_coverage_vector, training=training)

                # record previous word states
                decode_previous_word_states = word_states

                # [:, line_index, word_index]
                y_true = y_token_id[:, line_index, word_index]
                loss_weight = tf.ones(y_true.shape, tf.float32)
                if model.pointer and model.word_coverage and force_copy:
                    y_true, loss_weight = forced_copying_process(y_true, k_x_token_id, k_x_word_mask, k_sentence_context_w, word_coverage_vector, unk_num)

                # each time append [b], total (line_index - 1) * (out_seq_len - 1)
                if model.word_coverage:
                    # [b,top-k,seq_len,2]=>[b,top-k,seq_len]=>[b,top-k]=>[b]
                    word_penalty = word_penalty.write(word_index, tf.reduce_sum(tf.reduce_sum(tf.reduce_min(tf.concat([word_coverage_vector, word_context_weight], -1), -1), -1), -1))
                    word_coverage_vector = next_cover

                # record information about loss
                word_distribution = word_distribution.write(word_index, distribution)  # [b, class]
                word_y_true = word_y_true.write(word_index, y_true)  # [b]
                word_distribution_loss_weight = word_distribution_loss_weight.write(word_index, loss_weight)  # [b]

                # [b]
                out_word = out_word.write(word_index+1, tf.argmax(distribution, axis=-1, output_type=tf.int32))

            # [b, top-k, seq_len, 1] => [b, line, seq, 1]
            # tmp = tf.TensorArray(dtype=tf.float32, size=batch_size)
            # for b in range(batch_size):
            #     each_tensor = tf.squeeze(all_word_coverage_vector[b], -1)  # [line, seq]
            #     each_indices = tf.expand_dims(k_indices[b], -1)  # [line, 1]
            #     each_updates = tf.squeeze(word_coverage_vector[b], -1)  # [line, seq]
            #     tmp = tmp.write(b, tf.tensor_scatter_nd_update(each_tensor, each_indices, each_updates))
            # all_word_coverage_vector = tf.expand_dims(tmp.stack(), -1)

            all_word_coverage_vector = tf.tensor_scatter_nd_update(
                tf.squeeze(all_word_coverage_vector, -1),  # [batch_size, in_line_len, in_seq_len]
                k_indices,  # [batch_size, batch_size*top-k, 2]
                tf.squeeze(word_coverage_vector, -1)   # [batch_size, top-k, in_seq_len]
            )
            all_word_coverage_vector = tf.expand_dims(all_word_coverage_vector, -1)
            # def save_word_coverage(each):
            #     each_tensor = tf.squeeze(each[0], -1)
            #     each_indices = tf.expand_dims(each[1], -1)
            #     each_updates = tf.squeeze(each[2], -1)
            #     return tf.expand_dims(tf.tensor_scatter_nd_update(each_tensor, each_indices, each_updates), -1)
            #
            # # max_thread, min_thread from GRU_seq2seq
            # thread_num = max(min(max_thread, batch_size//4), min_thread)
            # pool = ThreadPool(thread_num)
            # all_word_coverage_vector = pool.map(save_word_coverage, zip(all_word_coverage_vector, k_indices, word_coverage_vector))
            # pool.close()
            # pool.join()
            # all_word_coverage_vector = tf.stack(all_word_coverage_vector, 0)

            # [b]*out_seq_len => [b, out_seq_len]
            out_word = tf.transpose(out_word.stack()[1:], [1, 0])
            out_sentence = out_sentence.write(line_index+1, out_word)

            # [b, out_seq_len] * (out_line_len)
            if model.word_coverage:
                word_coverage_penalty = word_coverage_penalty.write(line_index, tf.transpose(word_penalty.stack(), [1, 0]))
            all_distribution = all_distribution.write(line_index, tf.transpose(word_distribution.stack(), [1, 0, 2]))  # [b, out_seq_len, class]
            all_y_true = all_y_true.write(line_index, tf.transpose(word_y_true.stack(), [1, 0]))  # [b, out_seq_len]
            all_distribution_loss_weight = all_distribution_loss_weight.write(line_index, tf.transpose(word_distribution_loss_weight.stack(), [1, 0]))  # [b, out_seq_len]

        all_distribution = tf.transpose(all_distribution.stack(), [1, 0, 2, 3])  # [b, out_line_len, out_seq_len, class]
        all_y_true = tf.transpose(all_y_true.stack(), [1, 0, 2])  # [b, out_line_len, out_seq_len]
        all_distribution_loss_weight = tf.transpose(all_distribution_loss_weight.stack(), [1, 0, 2])  # [b, out_line_len, out_seq_len]

        y_word_effect = 1. - tf.cast(y_word_mask, tf.float32)
        # [b, out_line_len, out_seq_len] * [b, out_line_len, out_seq_len] ! 1e-10: prevent log(0) INF
        loss = losses.sparse_categorical_crossentropy(all_y_true, all_distribution + 1e-10) * all_distribution_loss_weight * y_word_effect
        loss = tf.reduce_sum(loss) / tf.reduce_sum(y_word_effect)

        if model.sentence_coverage:
            # [b, out_line_len]
            y_line_effect = 1. - tf.cast(y_line_mask, tf.float32)
            # [b, out_line_len]
            sentence_coverage_penalty = tf.transpose(sentence_coverage_penalty.stack(), [1, 0])
            sentence_cov_loss = tf.reduce_sum(sentence_coverage_penalty * y_line_effect) / tf.reduce_sum(y_line_effect)
            loss += sentence_lambda * sentence_cov_loss

        if model.word_coverage:
            # [b, out_line_len, out_seq_len]
            word_coverage_penalty = tf.transpose(word_coverage_penalty.stack(), [1, 0, 2])
            word_cov_loss = tf.reduce_sum(word_coverage_penalty * y_word_effect) / tf.reduce_sum(y_word_effect)
            loss += word_lambda * word_cov_loss
    if training:
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 4.)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # [b, out_seq_len] * out_line_len => [b, out_line_len, out_seq_len]
    out_sentence = tf.transpose(out_sentence.stack()[1:], [1, 0, 2])

    return loss, out_sentence


@tf.function
def hie_analysis_process(model: HierarchicalSeq2Seq, x_token_id, x_word_mask, x_line_mask, y_line_len, y_seq_len, y_first_line, y_first_id, unk_num=1):
    """
    analysis one code snippet

    :param model: HierarchicalSeq2Seq
    :param x_token_id: [b, in_line, in_seq]
    :param x_word_mask: [b, in_line, in_seq]
    :param x_line_mask: [b, in_line]
    :param y_line_len: Tensor(num)
    :param y_seq_len: Tensor(num)
    :param y_first_line: [out_seq]
    :param y_first_id: num
    :param unk_num: num
    :return: sequences, line_weight, word_weight, [vocab_distribution, pointer_distribution, p_gen]
    """
    batch_size = tf.shape(x_token_id)[0]
    in_line_len = tf.shape(x_token_id)[1]
    in_seq_len = tf.shape(x_token_id)[2]
    out_line_len = y_line_len
    out_seq_len = y_seq_len
    training = False

    input_x_token_id = x_token_id
    if model.pointer:
        input_x_token_id = tf.where(input_x_token_id < model.vocab_size, input_x_token_id,
                                    tf.fill([batch_size, in_line_len, in_seq_len], unk_num))
    # [b, line, word_seq, 2*hid], [word_layers, b, line, hid]
    word_encode_sequence, word_states = model.word_encode_process(input_x_token_id, training=training)
    # [b, line, word_layer*hid*2], [sentence_layer, b, word_layer*hid]
    sentence_encode_sequence, sentence_states = model.sentence_encode_process(word_states, training=training)

    # coverage
    sentence_coverage_vector = tf.zeros([batch_size, in_line_len, 1])
    all_word_coverage_vector = tf.zeros([batch_size, in_line_len, in_seq_len, 1])

    # sentences [b,seq]
    out_sentence = tf.TensorArray(dtype=tf.int32, size=out_line_len + 1)
    out_sentence = out_sentence.write(0, tf.broadcast_to(y_first_line, [batch_size, out_seq_len]))

    # weight
    all_line_weight = tf.TensorArray(dtype=tf.float32, size=out_line_len)
    all_inline_word_weight = tf.TensorArray(dtype=tf.float32, size=out_line_len)

    # distribution
    all_vocab_distribution = tf.TensorArray(dtype=tf.float32, size=out_line_len)
    all_pointer_distribution = tf.TensorArray(dtype=tf.float32, size=out_line_len)
    all_pgen_score = tf.TensorArray(dtype=tf.float32, size=out_line_len)

    decode_previous_word_states = tf.zeros([model.word_layer_num, batch_size, model.hidden])

    for line_index in range(out_line_len):
        previous_sentence = out_sentence.read(line_index)
        # if previous has some extra vocab tokens, restore to normal tokens
        if model.pointer:
            previous_sentence = tf.where(previous_sentence < model.vocab_size, previous_sentence,
                                         tf.fill([batch_size, out_seq_len], unk_num))

        word_states, sentence_states, sentence_context_weight, sentence_next_coverage_vector = \
            model.sentence_decode_process(previous_sentence, sentence_encode_sequence, x_line_mask, sentence_states,
                                          decode_previous_word_states, sentence_coverage_vector, training=training)

        all_line_weight = all_line_weight.write(line_index, sentence_context_weight)

        # top-k data preparation
        k_sentence_context_w, k_word_encode_seq, k_x_word_mask, k_x_token_id, k_indices = \
            model.before_word_decode_process(sentence_context_weight, word_encode_sequence, x_word_mask, x_token_id)

        # coverage
        if model.sentence_coverage:
            sentence_coverage_vector = sentence_next_coverage_vector
        word_coverage_vector = tf.gather_nd(all_word_coverage_vector, k_indices)  # [b,top-k,seq_len,1]

        # Prepare the first word
        out_word = tf.TensorArray(dtype=tf.int32, size=out_seq_len + 1)
        out_word = out_word.write(0, tf.fill([batch_size], y_first_id))

        # weight
        inline_word_weight = tf.TensorArray(dtype=tf.float32, size=out_seq_len)

        # inline distribution
        vocab_distribution = tf.TensorArray(dtype=tf.float32, size=out_seq_len)
        pointer_distribution = tf.TensorArray(dtype=tf.float32, size=out_seq_len)
        pgen_score = tf.TensorArray(dtype=tf.float32, size=out_seq_len)



        for word_index in range(out_seq_len):
            previous_word = out_word.read(word_index)
            # restore to normal token
            if model.pointer:
                previous_word = tf.where(previous_word < model.vocab_size, previous_word,
                                         tf.fill([batch_size], unk_num))
            previous_word = tf.expand_dims(previous_word, -1)

            distribution, word_states, word_context_weight, next_cover, distribution_list = \
                model.word_decode_process(previous_word, k_sentence_context_w, k_word_encode_seq, k_x_word_mask,
                                          word_states, k_x_token_id, word_coverage_vector, training=training)

            # record previous word states
            decode_previous_word_states = word_states

            # coverage
            if model.word_coverage:
                word_coverage_vector = next_cover

            inline_word_weight = inline_word_weight.write(word_index, word_context_weight)

            vocab_distribution = vocab_distribution.write(word_index, distribution_list[0])
            if model.pointer:
                pointer_distribution = pointer_distribution.write(word_index, distribution_list[1])
                pgen_score = pgen_score.write(word_index, distribution_list[2])

            # [b]
            out_word = out_word.write(word_index + 1, tf.argmax(distribution, axis=-1, output_type=tf.int32))

        # inline decode end

        # coverage vector update
        all_word_coverage_vector = tf.tensor_scatter_nd_update(
            tf.squeeze(all_word_coverage_vector, -1),  # [batch_size, in_line_len, in_seq_len]
            k_indices,  # [batch_size, batch_size*top-k, 2]
            tf.squeeze(word_coverage_vector, -1)   # [batch_size, top-k, in_seq_len]
        )
        all_word_coverage_vector = tf.expand_dims(all_word_coverage_vector, -1)

        # [b]*out_seq_len => [b, out_seq_len]
        out_word = tf.transpose(out_word.stack()[1:], [1, 0])
        out_sentence = out_sentence.write(line_index + 1, out_word)

        # weight: [b, top-k, in_seq_len, 1] * out_seq_len
        inline_word_weight = tf.transpose(inline_word_weight.stack(), [1, 0, 2, 3, 4])
        all_inline_word_weight = all_inline_word_weight.write(line_index, inline_word_weight)

        # distribution: [b, class], [b, class], [b, 1] * out_seq_len
        vocab_distribution = tf.transpose(vocab_distribution.stack(), [1, 0, 2])
        all_vocab_distribution = all_vocab_distribution.write(line_index, vocab_distribution)
        if model.pointer:
            pointer_distribution = tf.transpose(pointer_distribution.stack(), [1, 0, 2])
            pgen_score = tf.transpose(pgen_score.stack(), [1, 0, 2])
            all_pointer_distribution = all_pointer_distribution.write(line_index, pointer_distribution)
            all_pgen_score = all_pgen_score.write(line_index, pgen_score)

    # decode end

    # [b, out_line_len, out_seq_len]
    out_sentence = tf.transpose(out_sentence.stack()[1:], [1, 0, 2])
    # [b, out_line_len, in_line_len, 1]
    all_line_weight = tf.transpose(all_line_weight.stack(), [1, 0, 2, 3])
    # [b, out_line_len, out_seq_len, top_k, in_seq_len, 1]
    all_inline_word_weight = tf.transpose(all_inline_word_weight.stack(), [1, 0, 2, 3, 4, 5])
    distribution_list = []
    if model.pointer:
        # [b, out_line_len, out_seq_len, class]
        all_vocab_distribution = tf.transpose(all_vocab_distribution.stack(), [1, 0, 2, 3])
        # [b, out_line_len, out_seq_len, class]
        all_pointer_distribution = tf.transpose(all_pointer_distribution.stack(), [1, 0, 2, 3])
        # [b, out_line_len, out_seq_len, 1]
        all_pgen_score = tf.transpose(all_pgen_score.stack(), [1, 0, 2, 3])
        distribution_list = [all_vocab_distribution, all_pointer_distribution, all_pgen_score]

    return out_sentence, all_line_weight, all_inline_word_weight, distribution_list
