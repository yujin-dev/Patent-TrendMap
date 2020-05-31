import tensorflow as tf


class network:
    def __init__(self, setting, input_data, output_data):
        super(network, self).__init__(setting, input_data, output_data)

    def fc_weight(self, name, input_dim, output_dim):

        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        w = tf.get_variable(
            name=name,
            shape=[input_dim, output_dim],
            initializer=initializer,
            regularizer=regularizer,
        )
        b = tf.Variable(tf.random_normal([output_dim], stddev=0.01))
        return w, b

    def fc_layer(self, input, weight, bias, drop_out=None, final=False):

        layer = tf.matmul(input, weight)
        add_bias = tf.nn.bias_add(layer, bias)
        activation = tf.nn.relu(add_bias)
        if final == True:
            activation = add_bias

        if drop_out != None:
            drop_out = tf.nn.dropout(activation, self.keep_prob)
            return drop_out
        else:
            return activation

    def LSTM(self):

        with tf.variable_scope("Attention_Encode", reuse=tf.AUTO_REUSE):

            cell = tf.nn.rnn_cell.LSTMCell(self.macro_seq_len)
            first_outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
            print("encoding LSTM : ", first_outputs.shape)

            first_lstm_dimension = (
                first_outputs.get_shape().as_list()[1]
                * first_outputs.get_shape().as_list()[2]
            )
            lstm_outputs = tf.reshape(first_outputs, [-1, first_lstm_dimension])
            print("encoding output : ", lstm_outputs.shape)

            W, b = self.fc_weight(
                "e1", lstm_outputs.get_shape().as_list()[1], self.E[0]
            )
            fc_layer = self.fc_layer(lstm_outputs, W, b, drop_out=True)
            print(" == fc_layer1 == : ", fc_layer.shape)

            if len(self.E) > 1:
                W, b = self.fc_weight(
                    "e2", fc_layer.get_shape().as_list()[1], self.E[1]
                )
                fc_layer = self.fc_layer(fc_layer, W, b, drop_out=True)
                print(" == fc_layer2 == : ", fc_layer.shape)

            if len(self.E) > 2:
                W, b = self.fc_weight(
                    "e3", fc_layer.get_shape().as_list()[1], self.E[2]
                )
                fc_layer = self.fc_layer(fc_layer, W, b, drop_out=True)
                print(" == fc_layer3 == : ", fc_layer.shape)

            W, b = self.fc_weight(
                "e", fc_layer.get_shape().as_list()[1], first_lstm_dimension
            )
            fc_layer = self.fc_layer(fc_layer, W, b, drop_out=True)
            bn_layer = tf.contrib.layers.batch_norm(
                fc_layer, center=True, scale=True, is_training=self.phase, fused=False
            )
            attention_layer = tf.nn.softmax(bn_layer)
            print(" == attention layer == : ", attention_layer.shape)

        with tf.variable_scope("Decoding_Layer", reuse=tf.AUTO_REUSE):

            multiply_val = tf.multiply(lstm_outputs, attention_layer)
            multiply_val = tf.reshape(
                multiply_val, [-1, self.macro_seq_len, self.macro_seq_len]
            )
            print("attention weight to decoding : ", multiply_val.shape)

            cell_2 = tf.nn.rnn_cell.LSTMCell(self.macro_seq_len)
            second_outputs, states_second = tf.nn.dynamic_rnn(
                cell_2, multiply_val, dtype=tf.float32
            )
            print("decoding LSTM : ", second_outputs.shape)

            second_lstm_dimension = (
                second_outputs.get_shape().as_list()[1]
                * second_outputs.get_shape().as_list()[2]
            )
            second_lstm_outputs = tf.reshape(
                second_outputs, [-1, second_lstm_dimension]
            )
            print("decoding output : ", second_lstm_outputs.shape)

            W, b = self.fc_weight(
                "d1", second_lstm_outputs.get_shape().as_list()[1], self.D[0]
            )
            second_fc_layer = self.fc_layer(second_lstm_outputs, W, b, drop_out=True)
            print(" == fc_layer1 == : ", second_fc_layer.shape)

            if len(self.D) > 1:
                W, b = self.fc_weight(
                    "d2", second_fc_layer.get_shape().as_list()[1], self.D[1]
                )
                second_fc_layer = self.fc_layer(second_fc_layer, W, b, drop_out=True)
                print(" == fc_layer2 == : ", second_fc_layer.shape)

            if len(self.D) > 2:
                W, b = self.fc_weight(
                    "d3", second_fc_layer.get_shape().as_list()[1], self.D[2]
                )
                second_fc_layer = self.fc_layer(second_fc_layer, W, b, drop_out=True)
                print(" == fc_layer3 == : ", second_fc_layer.shape)

            W, b = self.fc_weight("d", second_fc_layer.get_shape().as_list()[1], 3)
            y_pred = self.fc_layer(second_fc_layer, W, b, final=True)
            print(" == final_layer == : ", y_pred.shape)

        return y_pred
