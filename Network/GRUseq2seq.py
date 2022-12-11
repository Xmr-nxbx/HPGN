import math
from re import A

import tensorflow as tf
from tensorflow.keras import *
from multiprocessing.dummy import Pool as ThreadPool

pad_flag = '<pad>'
vocab_pad_flag = '$p'
unk_id = 1
min_thread = 3
max_thread = 10


class GRUBiEncoder(layers.Layer):
    def __init__(self, hid_dim, layer_num, dropout=0.1):
        super(GRUBiEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.layer_num = layer_num

        self.gru_layers = [
            layers.Bidirectional(layers.GRU(hid_dim, return_sequences=True, return_state=True, dropout=dropout)) for _
            in range(self.layer_num)]

    def call(self, inputs, mask=None, training=None, *args, **kwargs):
        # inputs: [batch, seq, embed_dim]
        # mask: [batch, seq]
        # training: bool
        x = inputs
        batch_size = tf.shape(x)[0]
        forward_states = tf.TensorArray(dtype=tf.float32, size=self.layer_num)
        backward_states = tf.TensorArray(dtype=tf.float32, size=self.layer_num)
        initial_state = [tf.zeros([batch_size, tf.constant(self.hid_dim)])] * 2
        for i in range(self.layer_num):
            x, forward_s, backward_s = self.gru_layers[i](x,
                                                          initial_state=initial_state,
                                                          training=training, mask=mask)
            forward_states = forward_states.write(i, forward_s)
            backward_states = backward_states.write(i, backward_s)
        # x: [b, seq, hid*2]
        # states: layer_num * [batch, hid]
        return x, forward_states.stack(), backward_states.stack()


class BahdanauaAttention(layers.Layer):
    def __init__(self, hid_dim, coverage=False):
        super(BahdanauaAttention, self).__init__()
        self.hid_dim = hid_dim
        self.coverage = coverage

        self.W_h = layers.Dense(hid_dim, use_bias=False)
        self.W_s = layers.Dense(hid_dim)
        if coverage:
            self.W_c = layers.Dense(1, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)

    def call(self, key, query, mask=None, cover_vec=None):
        # key: [b, 1, hid]
        # query: [b, seq, 2*hid]
        # mask: [b, seq] pad: 1
        # cover_vec: [b, seq, 1] sum of all previous decoder time-step

        # [b, seq, hid]
        x = self.W_h(query) + self.W_s(key)
        if cover_vec is None:
            cover_vec = tf.expand_dims(tf.zeros(tf.shape(mask)), -1)

        x += self.W_c(cover_vec)
        # [b, seq, 1]
        x = self.V(tf.nn.tanh(x))
        if mask is not None:
            # [b, seq, 1]
            # pad_place = (1. - tf.cast(tf.expand_dims(mask, -1), dtype=x.dtype))
            pad_place = tf.cast(tf.expand_dims(mask, -1), dtype=tf.float32)
            x += pad_place * -1e10
        # [b, seq, 1]
        context_weight = tf.nn.softmax(x, axis=-2)
        # [b, 1, 2*hid]
        context_vector = tf.reduce_sum(context_weight * query, axis=-2, keepdims=True)
        # calc next coverage_vector

        cover_vec += context_weight

        # [b, 1, 2*hid], [b, seq, 1] [b, seq, 1]
        return context_vector, context_weight, cover_vec


def initial_coverage_vector(batch_size, encode_length):
    return tf.zeros((batch_size, encode_length, 1), tf.float32)


class GRUDecoder(layers.Layer):
    def __init__(self, hid_dim, layer_num, dropout=0.1, coverage=False):
        super(GRUDecoder, self).__init__()
        self.hid_dim = hid_dim
        self.layer_num = layer_num
        self.coverage = coverage

        self.attention = BahdanauaAttention(hid_dim, coverage)
        self.gru_layers = [layers.GRU(self.hid_dim, dropout=dropout, return_state=True, return_sequences=True) for _ in
                           range(layer_num)]
        self.fc = layers.Dense(hid_dim)

    def decode_from_multi_line_process(self, decode_input, encode_sequence, encode_mask, previous_state, coverage_vector=None, line_weight=None, training=None):
        # decode_input: [b, 1, hid]
        # encode_sequence: [b, line, seq, 2*hid]
        # encode_mask: [b, line, seq]   pad: 1
        # previous_state: [b, hid] from Bi-encode states concatenation or previous
        # coverage_vector: [b, line, seq, 1]
        # line_weight: [b, line]
        batch = tf.shape(encode_sequence)[0]
        line = tf.shape(encode_sequence)[1]
        seq = tf.shape(encode_sequence)[2]
        hid = tf.shape(encode_sequence)[3]

        x = tf.cast(decode_input, tf.float32)
        previous_state = tf.cast(previous_state, tf.float32)
        decode_states_array = tf.TensorArray(dtype=tf.float32, size=self.layer_num)
        for i in range(self.layer_num):
            x, state = self.gru_layers[i](x, initial_state=previous_state[i], training=training)
            decode_states_array = decode_states_array.write(i, state)
        decode_states = decode_states_array.stack()

        # context_vec_list = tf.TensorArray(dtype=tf.float32, size=tf.shape(line_weight)[1])
        # context_weight_list = tf.TensorArray(dtype=tf.float32, size=tf.shape(line_weight)[1])
        # # [b, 1, 1, line]
        # line_weight = tf.expand_dims(tf.expand_dims(line_weight, -2), -2)
        # for i in range(tf.shape(line_weight)[-1]):
        #     c_v, c_w, _ = self.attention(x, encode_sequence[:, i], encode_mask[:, i], coverage_vector[:, i])
        #     context_vec_list = context_vec_list.write(i, c_v * line_weight[..., i])
        #     context_weight_list = context_weight_list.write(i, c_w * line_weight[..., i])
        #
        # context_vec_list = tf.transpose(context_vec_list.stack(), [1, 2, 3, 0])  # [b, 1, 2*hid, line]
        # context_weight = tf.transpose(context_weight_list.stack(), [1, 0, 2, 3])  # [b, line, seq, 1]
        # context_vec = tf.reduce_sum(context_vec_list, -1)  # [b, 1, 2*hid]
        # # next_coverage and context_weight
        # next_coverage_vector = coverage_vector + context_weight  # [b, line, seq, 1]

        encode_sequence = tf.reshape(encode_sequence, [batch, line*seq, hid])
        encode_mask = tf.reshape(encode_mask, [batch, line*seq])
        cover_v = tf.reshape(coverage_vector, [batch, line*seq, 1])

        context_vector, context_weight, next_coverage_vector = self.attention(x, encode_sequence, encode_mask, cover_v)

        context_weight = tf.reshape(context_weight, [batch, line, seq, 1])
        next_coverage_vector = tf.reshape(next_coverage_vector, [batch, line, seq, 1])

        # [b, 1, 3 * hid]
        x = tf.concat([x, context_vector], axis=-1)
        # [b, 1, hid]
        x = self.fc(x)

        # x: [b, 1, hid]
        # decode_states: [layer, b, hid]
        # context_vector: [b, 1, 2*hid] for pointer-generator
        # context_weight: [b, line, seq, 1]
        # coverage_vector: [b, line, seq, 1]
        return x, decode_states, context_vector, context_weight, next_coverage_vector

    def call(self, decode_input, encode_sequence, encode_mask, previous_state, coverage_vector=None, training=None):
        # decode_input: [b, 1, hid]
        # encode_sequence: [b, seq, 2*hid] or [b, line, seq, 2*hid]
        # encode_mask: [b, seq] or [b, line, seq]   pad: 1
        # previous_state: [b, hid] from Bi-encode states concatenation or previous
        # coverage_vector: [b, seq, 1]

        x = tf.cast(decode_input, tf.float32)
        previous_state = tf.cast(previous_state, tf.float32)
        decode_states_array = tf.TensorArray(dtype=tf.float32, size=self.layer_num)
        for i in range(self.layer_num):
            x, state = self.gru_layers[i](x, initial_state=previous_state[i], training=training)
            decode_states_array = decode_states_array.write(i, state)
        decode_states = decode_states_array.stack()

        # [b, 1, 2*hid], [b, seq, 1] [b, seq, 1]
        context_vec, context_weight, next_coverage_vector = self.attention(x, encode_sequence, encode_mask,
                                                                               coverage_vector)

        # [b, 1, 3 * hid]
        x = tf.concat([x, context_vec], axis=-1)
        # [b, 1, hid]
        x = self.fc(x)

        # x: [b, 1, hid]
        # decode_states: [layer, b, hid]
        # context_vector: [b, 1, 2*hid] for pointer-generator
        # context_weight: [b, seq, 1] or [b, line, seq, 1]
        # coverage_vector: [b, seq, 1] or [b, line, seq, 1]
        return x, decode_states, context_vec, context_weight, next_coverage_vector

        # if self.coverage:
        #     return x, decode_states, context_vec, context_weight, next_coverage_vector
        # return x, decode_states, context_vec, context_weight, None


# step 1: 数据准备阶段
def get_extra_vocab(raw_source, token):
    """
    from token generate extra_vocab

    :param raw_source: [b, seq] or [b, line, seq], encoder part, for generate extra token
    :param token: [class] for generate extra token
    :return extra_vocab: same as raw_source
    """
    if not tf.is_tensor(raw_source):
        raw_source = tf.convert_to_tensor(raw_source)

    batch_size = tf.shape(raw_source)[0]
    seq_len = tf.shape(raw_source)[-1]
    expand_flag = False
    if tf.rank(raw_source) == 3:
        expand_flag = True
        raw_source = tf.reshape(raw_source, [-1, seq_len])

    # due to video card memory
    def compromise():
        def build_mask(batch_data):
            batch_source = batch_data
            indices = tf.where(tf.expand_dims(batch_source, -1) == token)
            # [b, seq] token not in vocab: 1
            mask = tf.scatter_nd(indices=indices[:, :-1], updates=tf.ones([indices.shape[0]], tf.int32), shape=batch_source.shape)
            mask = tf.logical_not(tf.cast(mask, tf.bool))
            return mask
        if raw_source.shape[0] < 4000 and seq_len < 500:
            return build_mask(raw_source)
        else:
            steps = math.ceil(raw_source.shape[0] / 1000)
            mask_list = [build_mask(raw_source[1000*s:1000*(s+1)]) for s in range(steps)]
            return tf.concat(mask_list, 0)

    mask = compromise()

    def speed_up():
        def build_line_token(batch_data):
            source_line, mask_line = batch_data
            selection = tf.boolean_mask(source_line, mask_line)
            line_token = tf.unique(selection).y
            return line_token

        thread_num = max(min(max_thread, raw_source.shape[0]//4), min_thread)

        extra_vocab_list = []
        for i in range(len(raw_source)):
            extra_vocab_list.append(build_line_token([raw_source[i], mask[i]]))
        # pool = ThreadPool(thread_num)
        # extra_vocab_list = pool.map(build_line_token, zip(raw_source.numpy().tolist(), mask.numpy().tolist()))
        # pool.close()
        # pool.join()
        return tf.keras.preprocessing.sequence.pad_sequences(extra_vocab_list, seq_len, object, 'post', 'post', vocab_pad_flag)

    extra_vocab = speed_up()

    if expand_flag:
        return tf.reshape(extra_vocab, [batch_size, -1, seq_len])

    return extra_vocab


# step 2: 当前batch训练阶段
def get_concat_vocab(extra_vocab, token):
    """
    from token generate concat_vocab

    :param extra_vocab: [b, seq] or [b, top-k, seq]
    :param token: [class]
    :return: concat_vocab: [b, class + top-k * seq]
    """
    if not tf.is_tensor(extra_vocab):
        extra_vocab = tf.convert_to_tensor(extra_vocab)

    batch_size = tf.shape(extra_vocab)[0]
    class_num = tf.shape(token)[0]
    if tf.rank(extra_vocab) == 3:
        top_k = tf.shape(extra_vocab)[1]
        seq_len = tf.shape(extra_vocab)[2]

        mask = extra_vocab != vocab_pad_flag

        def speed_up():
            def build_extra_vocab(batch_data):
                batch_vocab, batch_mask = batch_data
                return tf.unique(tf.boolean_mask(batch_vocab, batch_mask)).y
            thread_num = max(min(max_thread, batch_size // 4), min_thread)
            pool = ThreadPool(thread_num)
            new_vocab = pool.map(build_extra_vocab, zip(extra_vocab.numpy(), mask.numpy()))
            pool.close()
            pool.join()
            return new_vocab
        vocab = speed_up()

        extra_vocab = tf.keras.preprocessing.sequence.pad_sequences(vocab, maxlen=top_k * seq_len, dtype=object,
                                                                    padding='post', truncating='post',
                                                                    value=vocab_pad_flag)

    concat_vocab = tf.concat([tf.broadcast_to(tf.expand_dims(token, 0), (batch_size, class_num)), extra_vocab], axis=-1)

    return concat_vocab


# step 3: 当前batch训练阶段-权重定位、输出label定位
def get_token_id_from_concat_vocab(sequences, vocab, unk=unk_id):
    """
    from vocab generate sequences_id

    :param sequences: [b, seq] or [b, line, seq]
    :param vocab: [b, class]
    :param unk: Unknown token id
    :return: sequences_id: same as input
    """
    if not tf.is_tensor(sequences):
        sequences = tf.convert_to_tensor(sequences)

    batch_size = tf.shape(sequences)[0]
    seq_len = tf.shape(sequences)[-1]
    expand_flag = False
    if tf.rank(sequences) == 3:
        expand_flag = True
        sequences = tf.reshape(sequences, [batch_size, -1])

    def speed_up():
        def map_func(inputs):
            # inputs[0]: [seq_len * top_k]
            # inputs[1]: [class]
            # [word_index, word_id]
            extend_seq_len = tf.shape(inputs[0])[-1]
            indices = tf.cast(tf.where(tf.expand_dims(inputs[0], -1) == inputs[1]), tf.int32)
            # 单词在单词表中查找，不会重复
            word_indices = indices[:, 0:1]
            word_id = indices[:, 1]
            sequence_id = tf.tensor_scatter_nd_update(tf.fill([extend_seq_len], unk), word_indices, word_id)
            return sequence_id

        thread_num = max(min(max_thread, batch_size // 4), min_thread)
        sequences_id = []
        for i in range(len(sequences)):
            sequences_id.append(map_func([sequences[i], vocab[i]]))
        # pool = ThreadPool(thread_num)
        # sequences_id = pool.map(map_func, zip(sequences.numpy(), vocab.numpy()))
        # pool.close()
        # pool.join()
        return sequences_id

    sequences_id = speed_up()

    if expand_flag:
        return tf.reshape(sequences_id, [batch_size, -1, seq_len])
    return tf.convert_to_tensor(sequences_id)


# step 4: 计算权重之和
def get_distribution_from_both(source_id, source_mask, attention_weight, vocab_distribution, p_gen):
    """
    pointer-generator distribution generate step

    :param source_id: [b, seq]
    :param source_mask: [b, seq] pad: 1
    :param attention_weight: [b, seq, 1]
    :param vocab_distribution: [b, class]
    :param p_gen: [b, 1]
    :return: [b, class + seq]
    """
    batch_size = tf.shape(source_id)[0]
    attention_weight = tf.squeeze(attention_weight, -1)

    # if tf.equal(tf.rank(source_id), 3):
    #     source_id = tf.reshape(source_id, [batch_size, -1])  # [b, seq]
    #     source_mask = tf.reshape(source_mask, [batch_size, -1])  # [b, seq]
    #     attention_weight = tf.reshape(attention_weight, [batch_size, -1])  # [b, seq]

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
    final_distribution = vocab_distribution + attention_distribution.stack()

    return final_distribution


# step 5: 当前batch的token输出
def get_token_from_vocab(sequences_id, vocab):
    """
    from sequences_id generate token

    :param sequences_id: [b, line, seq] or [b, seq]
    :param vocab: [b, class] or [class]
    :return: sequences same as input
    """
    if not tf.is_tensor(sequences_id):
        sequences_id = tf.convert_to_tensor(sequences_id)
    if not tf.is_tensor(vocab):
        vocab = tf.convert_to_tensor(vocab)

    batch_size = tf.shape(sequences_id)[0]
    seq_len = tf.shape(sequences_id)[-1]
    expand_flag = False
    if tf.rank(sequences_id) == 3:
        expand_flag = True
        sequences_id = tf.reshape(sequences_id, [batch_size, -1])
    if tf.rank(vocab) == 1:
        vocab = tf.broadcast_to(vocab, [batch_size, tf.shape(vocab)[0]])

    def speed_up():
        def build_sequence(batch_data):
            batch_vocab, batch_seq_id = batch_data
            return tf.gather_nd(params=batch_vocab, indices=tf.expand_dims(batch_seq_id, -1))

        thread_num = max(min(max_thread, batch_size // 4), min_thread)
        pool = ThreadPool(thread_num)
        sequences = pool.map(build_sequence, zip(vocab.numpy(), sequences_id.numpy()))
        pool.close()
        pool.join()
        return sequences
    sequences = speed_up()

    if expand_flag:
        return tf.reshape(sequences, [batch_size, -1, seq_len])
    return tf.convert_to_tensor(sequences)


class PointerGenerator(layers.Layer):
    def __init__(self):
        super(PointerGenerator, self).__init__()
        self.W_h = layers.Dense(1, use_bias=False)
        self.W_s = layers.Dense(1, use_bias=False)
        self.W_x = layers.Dense(1)

    def call(self, context_vector, decoder_state, decoder_input):
        # [b, 2*hid] [b, hid] [b, embed] => [b, 1]
        return tf.nn.sigmoid(self.W_h(context_vector) + self.W_s(decoder_state) + self.W_x(decoder_input))


class PointerGeneratorSeq2Seq(Model):
    def __init__(self, vocab_size, layer_num, hidden, dropout=0.1, pointer=True, coverage=True):
        super(PointerGeneratorSeq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.layer_num = layer_num
        self.hidden = hidden
        self.pointer = pointer
        self.coverage = coverage

        self.embedding = layers.Embedding(vocab_size, hidden)
        self.encoder = GRUBiEncoder(hidden, layer_num, dropout)
        self.decoder = GRUDecoder(hidden, layer_num, dropout, coverage)
        self.reduce_weight = [layers.Dense(hidden, activation='relu') for _ in range(layer_num)]
        self.fc = layers.Dense(vocab_size, activation=tf.nn.softmax, use_bias=False)

        if pointer:
            self.pointer = PointerGenerator()

    def embedding_token(self, x):
        """
        return x embedding vector

        :param x: [b, seq]
        :return: embed_x [b, seq, hid]
        """
        return self.embedding(x)

    # step 1
    def encode_process(self, x_id, training=None):
        """
        return hidden sequence and state

        :param x_id: [b, seq]
        :param training: bool, for dropout
        :return:
        """
        # [b, seq, hid]
        x_embed = self.embedding_token(x_id)

        # [b, seq, hid*2], [layer_num, batch, hid] * 2
        x_sequence, x_forward_state, x_backward_state = self.encoder(x_embed, training=training)
        # [layer_num, b, hid * 2]
        state = tf.concat([x_forward_state, x_backward_state], axis=-1)
        x_state = tf.TensorArray(dtype=tf.float32, size=self.layer_num)

        for i in range(self.layer_num):
            reduced_state = self.reduce_weight[i](state[i])
            x_state = x_state.write(i, reduced_state)

        # [b, seq, hid*2]
        # [layer_num, batch, hid]
        return x_sequence, x_state.stack()

    # step 2
    def decode_process(self, y_id, x_sequence, x_mask, state, x_token_id, coverage_vector=None, training=None):
        """

        :param y_id: [b, 1]
        :param x_sequence: [b, seq, hid*2]
        :param x_mask: [b, x_seq]
        :param state: [layers, b, hid]
        :param x_token_id: [b, x_seq]
        :param coverage_vector: [b, x_seq, 1] or None
        :param training: bool, for dropout
        :return: distribution, state, context_weight, next_coverage
        """

        # [b, 1, hid]
        y_embed = self.embedding_token(y_id)
        # y, decode_state_list, context_vec, context_weight, [next_coverage_vector]
        y_out, state, context_vec, context_weight, next_coverage = self.decoder(y_embed, x_sequence, x_mask, state,
                                                                                coverage_vector=coverage_vector,
                                                                                training=training)
        # [b, 1, class] => [b, class]
        distribution = tf.squeeze(self.fc(y_out), -2)
        if self.pointer:
            decoder_state = state[-1]
            context_vec = tf.squeeze(context_vec, -2)
            y_embed = tf.squeeze(y_embed, -2)
            p_gen = self.pointer(context_vec, decoder_state, y_embed)
            distribution = get_distribution_from_both(x_token_id, x_mask, context_weight, distribution, p_gen)

        return distribution, state, context_weight, next_coverage

    def call(self, inputs, training=None, mask=None):
        pass


def pgn_before_send_data_process(dataset, flags, pad_unk_cl, token_limits, vocab_token, pointer=True):
    dataset = [tf.sparse.to_dense(each) for each in dataset]
    batch_size = tf.shape(dataset[0])[0]
    starts = [each[0] for each in flags]
    ends = [each[1] for each in flags]

    def convert_to_standard_input(index, add_flag):
        # [batch, line * word]
        data = tf.reshape(dataset[index], [batch_size, -1]).numpy().tolist()
        # [batch, tokens]
        if add_flag:
            data = [
                [bytes.decode(each_token) for each_token in each_batch if len(each_token) != 0] + [ends[index]]
                for each_batch in data
            ]
        else:
            data = [
                [bytes.decode(each_token) for each_token in each_batch if len(each_token) != 0]
                for each_batch in data
            ]
        return tf.keras.preprocessing.sequence.pad_sequences(data, token_limits[index], object, 'post', 'post', pad_unk_cl[0])
    dataset = [convert_to_standard_input(0, False), convert_to_standard_input(1, True)]

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
    token_mask_list = [tf.cast(each == pad_num, tf.int32) for each in token_id_list]
    # print(dataset[0][0])
    # print(token_mask_list[0][0])
    # print(dataset[1][0])
    # print(token_mask_list[1][0])

    concat_vocab = vocab_token
    if pointer:
        extra_vocab = get_extra_vocab(dataset[0], vocab_token)
        concat_vocab = get_concat_vocab(extra_vocab, vocab_token)
        token_id_list = [get_token_id_from_concat_vocab(each, concat_vocab, unk_num) for each in dataset]
    return token_id_list, token_mask_list, concat_vocab


@tf.function
def pgn_auto_regressive_process(model: PointerGeneratorSeq2Seq, optimizer, x_id, y_id, x_mask, y_mask, y_first_id, unk_id, training=None):
    """
    pointer-generator implementation

    :param model: PointerGeneratorSeq2Seq
    :param optimizer: tf.keras.optimizers
    :param x_id: [b, in_seq]
    :param y_id: [b, out_seq]
    :param x_mask: [b, in_seq] pad: 1
    :param y_mask: [b, out_seq] pad: 1
    :param y_first_id: number
    :param unk_id: number
    :param training: bool
    :return: loss, grads, out_sentence
    """
    batch_size = tf.shape(x_id)[0]
    in_seq_len = tf.shape(x_id)[1]
    out_seq_len = tf.shape(y_id)[1]
    cov_lambda = 1.

    with tf.GradientTape() as tape:
        x_encode_id = x_id
        if model.pointer:
            x_encode_id = tf.where(x_encode_id < model.vocab_size, x_encode_id, tf.fill(x_encode_id.shape, unk_id))
        # [b, in_seq, hidden*2], [layer, b, hidden]
        x_sequence, state = model.encode_process(x_encode_id, training=training)

        coverage_vector = initial_coverage_vector(batch_size, in_seq_len)
        out_sentence = tf.TensorArray(dtype=tf.int32, size=out_seq_len+1)
        out_sentence = out_sentence.write(0, tf.fill([batch_size], y_first_id))
        coverage_penalty = tf.TensorArray(dtype=tf.float32, size=out_seq_len)
        all_distribution = tf.TensorArray(dtype=tf.float32, size=out_seq_len)

        for i in range(out_seq_len):
            y_decode_id = tf.expand_dims(out_sentence.read(i), -1)
            if model.pointer:
                y_decode_id = tf.where(y_decode_id < model.vocab_size, y_decode_id, tf.fill(y_decode_id.shape, unk_id))
            # [b, class], [layer, b, hidden], [b, in_seq, 1], [b, in_seq, 1]
            distribution, state, context_weight, next_coverage_vector = model.decode_process(y_decode_id, x_sequence, x_mask, state, x_id, coverage_vector, training=training)

            if model.coverage:
                # [b, in_seq, 2] => [b, in_seq] => [b]
                coverage_penalty = coverage_penalty.write(i, tf.reduce_sum(tf.reduce_min(tf.concat([coverage_vector, context_weight], -1), -1), -1))
                coverage_vector = next_coverage_vector

            all_distribution = all_distribution.write(i, distribution)
            out_sentence = out_sentence.write(i + 1, tf.argmax(distribution, -1, tf.int32))

        # [b, out_seq_len]
        out_sentence = tf.transpose(out_sentence.stack()[1:], [1, 0])
        # [b, out_seq_len, class]
        all_distribution = tf.transpose(all_distribution.stack(), [1, 0, 2])
        # [b, out_seq_len]
        y_effective_places = tf.cast(1 - y_mask, tf.float32)

        # [b, out_seq_len]  1e-10: prevent log(0) INF
        loss = losses.sparse_categorical_crossentropy(y_id, all_distribution + 1e-10)
        # [b]
        loss = tf.reduce_sum(loss * y_effective_places) / tf.reduce_sum(y_effective_places)

        if model.coverage:
            # [b, out_seq_len]
            coverage_penalty = tf.transpose(coverage_penalty.stack(), [1, 0])

            cov_loss = tf.reduce_sum(coverage_penalty * y_effective_places) / tf.reduce_sum(y_effective_places)
            loss += cov_loss * cov_lambda
    if training:
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 4.)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, out_sentence



def mytest_GRUEncoder():
    batch = 3
    seq = 10
    embed = 2
    hid = 2
    layers = 1
    x = tf.random.normal([batch, seq, embed])
    mask = tf.cast(tf.random.uniform([batch, seq], 0, 2, tf.int32), tf.bool)
    encoder = GRUBiEncoder(hid, layers, dropout=0.2)
    output, f_states, b_states = encoder(x, mask=mask, training=True)
    # [b, seq, 2*hid]
    print(mask)
    print(output)
    # layers * [b, hid]
    print(len(f_states), len(b_states), f_states[-1].shape)


# mytest_GRUEncoder()

def mytest_Attention():
    hid = 20
    seq = 10
    attention = BahdanauaAttention(20, coverage=False)
    key = tf.random.normal((2, 1, hid))
    query = tf.random.normal((2, seq, hid))
    mask = tf.random.uniform((2, seq), 0, 2, tf.int32)
    cov_vec = tf.zeros((2, seq, 1))
    con_vec, con_wei, cov_vec = attention(key, query, mask, cov_vec)
    print(con_vec.shape)
    print(con_wei)
    print(cov_vec)


# mytest_Attention()

def mytest_GRUDecoder():
    layer = 2
    b = 3
    seq = 10
    hid = 4
    decoder = GRUDecoder(hid, layer, coverage=True)
    decode_input = tf.random.normal((b, 1, hid))
    encode_sequence = tf.random.normal((b, seq, hid))
    encode_mask = tf.random.uniform((b, seq), 0, 2, tf.int32)
    state = [tf.zeros((b, hid)) for _ in range(layer)]
    coverage = initial_coverage_vector(b, seq)
    logits, decode_state_list, context_vec, context_weight, next_coverage_vector = decoder(decode_input,
                                                                                           encode_sequence, encode_mask,
                                                                                           state, coverage)
    print(encode_mask)
    print(context_weight)


# mytest_GRUDecoder()


def mytest_get_extra_vocab():
    raw_source = [
                     [list('hello world'), list('this is my first time to study tensorflow')],
                     [list('my first mission is write seq2seq'), list('It\'s quite difficult')]
                 ] * (12000-1)
    for i in range(len(raw_source)):
        raw_source[i] = tf.keras.preprocessing.sequence.pad_sequences(raw_source[i], maxlen=500, dtype=object,
                                                                      padding='post', truncating='post', value=pad_flag)
    raw_source = tf.convert_to_tensor(raw_source, dtype=tf.string)
    token = tf.unique(list('sfdbuhrvbih')).y
    # [4, 40]
    # inputs = raw_source[:, 0]
    inputs = raw_source
    import time
    print('inputs', inputs.shape)
    start = time.time()
    extra_vocab = get_extra_vocab(inputs, token)
    now = time.time()
    print(extra_vocab.shape, now - start)
    start = now
    concat_vocab = get_concat_vocab(extra_vocab, token)
    now = time.time()
    print(concat_vocab.shape, now - start)
    start = now
    source_id = get_token_id_from_concat_vocab(inputs, concat_vocab)
    now = time.time()
    print(source_id.shape, now - start)
    start = now
    source = get_token_from_vocab(source_id, concat_vocab)
    now = time.time()
    print(source.shape, now - start)
    start = now

    p_gen = tf.zeros([inputs.shape[0], 1])
    attention_weight = tf.random.uniform(inputs.shape + [1])
    source_mask = tf.random.uniform(source_id.shape, 0, 2, tf.int32)
    final = get_distribution_from_both(source_id, source_mask, attention_weight,
                                       tf.ones((inputs.shape[0], token.shape[0])), p_gen)
    print(time.time() - start)
    print('result:')
    print(inputs[0])
    print(source_id[0])
    print(concat_vocab[0])
    print(source[0])
    print(source_mask[0].shape)
    print(attention_weight[0].shape)
    print(final[0].shape)


# mytest_get_extra_vocab()


def mytest_gruseq2seq():
    decoder_hid = 64
    encoder_hid = 32
    vocab_dim = 64
    vocab_size = 50
    layer = 1
    in_seq = 120
    out_seq = 30
    loss_lambda = 1
    embedding = layers.Embedding(vocab_size, vocab_dim)
    fc_out = layers.Dense(vocab_size, use_bias=False)
    encoder = GRUBiEncoder(encoder_hid, layer)
    ln1 = layers.LayerNormalization(trainable=True)
    ln2 = layers.LayerNormalization(trainable=True)
    decoder = GRUDecoder(decoder_hid, layer, coverage=True)
    pointer = PointerGenerator()
    optimizer = optimizers.Adam(0.005)

    inputs = [list(
        '据发布会消息，截至9月14日11时，厦门累计报告确诊病例35例，其中普通型22例，轻型13例，无症状感染者1例，新增的病例均为厦门现有确诊病例的密切接触者，即厦门首例确诊病例吴某某同一工厂的同事，新增的无症状感染者是确诊病例12，也就是第一医院外包服务人员蔡某某的邻居。'),
              list('据发布会通报，9月10日至9月14日8时，福建省累计报告本土新冠病毒阳性感染者139例，其中确诊病例120例（莆田市75例、泉州市12例、厦门市33例），无症状感染者19例（均在莆田市）。'),
              list(
                  '9月14日，外交部发言人赵立坚主持例行记者会。有记者提问，日本自民党总裁选举进入倒计时，候选人频频提及和中国有关的内容。13日，日本前外相、自民党总裁候选人岸田文雄再次表示要对抗中国，将在阁僚中新增负责人权和经济安保问题的相关职位。')
              ]*20
    labels = [list('厦门累计报告确诊35例，新增均为首例确诊的同事'),
              list('刚刚通报！泉州累计报告本土确诊12例！'),
              list('日本自民党总裁候选人炒作靖国神社话题，外交部驳斥')
              ]*20
    words = []
    vocab = {0: '<pad>', 1: '<unk>', 2: '<s>', 3: '</s>'}
    for i in range(len(inputs)):
        words.extend(inputs[i])
        words.extend(labels[i])
    import collections
    common = collections.Counter(words).most_common(vocab_size - len(vocab))
    for t, _ in common:
        vocab[len(vocab)] = t
    token = tf.convert_to_tensor(list(vocab.values()))

    for i in range(len(inputs)):
        inputs[i] = ['<s>'] + inputs[i] + ['</s>']
        labels[i] = ['<s>'] + labels[i] + ['</s>']
    inputs_token = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=in_seq, dtype=object,
                                                                 padding='post', truncating='post', value='<pad>')
    labels_token = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=out_seq, dtype=object,
                                                                 padding='post', truncating='post', value='<pad>')

    def get_token_id(sequence):
        lines = []
        for each in sequence:
            indices = tf.cast(tf.where(tf.expand_dims(each, -1) == token), tf.int32)
            word_indices = indices[:, 0:1]
            word_id = indices[:, 1]
            lines.append(tf.tensor_scatter_nd_update(tf.fill(each.shape, 1), word_indices, word_id))
        return tf.stack(lines)

    x = get_token_id(inputs_token)
    y = get_token_id(labels_token)
    # pad: 1 other: 0
    x_mask = tf.cast(x == 0, tf.float32)
    for step in range(50):
        with tf.GradientTape() as tape:
            embed_x = embedding(x)
            embed_y = embedding(y)
            sequence, forward_state, backward_state = encoder(embed_x, training=True)
            forward_state = ln1(forward_state, training=True)
            backward_state = ln2(backward_state, training=True)
            state = tf.concat([forward_state, backward_state], axis=-1)
            coverage_vec = initial_coverage_vector(x.shape[0], in_seq)

            logits_list = []
            coverage_penalty = []
            batch_size = x.shape[0]
            output = [tf.fill([batch_size, 1], 2)]

            for i in range(out_seq - 1):
                # Note: this coverage has been add context_weight
                # decode_embed = embedding(y[:, i:i+1])
                decode_embed = embedding(output[-1])
                out, state, context_vector, context_weight, next_coverage = decoder(decode_embed,
                                                                                    sequence,
                                                                                    x_mask,
                                                                                    state,
                                                                                    coverage_vec, training=True)
                # ====== test coverage_vec and next_coverage======
                mask = coverage_vec + context_weight != next_coverage
                if True in mask.numpy():
                    print(tf.boolean_mask(next_coverage, mask))
                    print(tf.boolean_mask(coverage_vec + context_weight, mask))
                # ====== end test ======
                # s_t = state[-1]
                p_gen = pointer(context_vector, out, embed_y[:, i:i + 1])
                # one by one
                logits = fc_out(out)
                logits = tf.nn.softmax(logits)
                logits_list.append(logits)
                output.append(tf.argmax(logits, -1, tf.int32))
                coverage_penalty.append(
                    tf.reduce_sum(tf.reduce_min(tf.concat([context_weight, coverage_vec], axis=-1), axis=-1),
                                   -1))  # [b, seq]
                coverage_vec = next_coverage
            output = tf.concat(output, -1)
            # [b, seq-1, class]
            logits_list = tf.concat(logits_list, axis=-2)
            # [b, seq-1]
            # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[:, 1:], logits=logits_list)
            loss = losses.categorical_crossentropy(tf.one_hot(y[:, 1:], vocab_size), logits_list)
            loss = tf.reduce_mean(loss) + loss_lambda * tf.reduce_mean(coverage_penalty)

        variables = embedding.trainable_variables + fc_out.trainable_variables + encoder.trainable_variables + decoder.trainable_variables + ln1.trainable_variables + ln2.trainable_variables + pointer.trainable_variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        print('step loss:', step, loss.numpy())
        if step % 10 == 0:
            for i in range(3):
                print(''.join([bytes.decode(token[w_i].numpy()) for w_i in output[i]]))

# mytest_gruseq2seq()