import tensorflow as tf
import numpy as np
import pandas as pd
from LSTM_classify import network
import os
from sklearn.preprocessing import minmax_scale


class model_run(network):
    def __init__(self, setting, macro_input, term_target, dir):

        macro_input.index = pd.DatetimeIndex(macro_input.index)
        self.macro_data = macro_input.fillna(0)
        term_target.index = pd.DatetimeIndex(term_target.index)
        self.term_target = term_target
        print(len(self.term_target))
        self.macro_seq_len = setting["Seq"]  # + setting['predict_term'] - 1
        self.macro_feature = self.macro_data.shape[1]
        self.setting = setting
        key = list(setting.keys())[0]
        self.learning = setting[key]
        self.E, self.D = setting["E_node"], setting["D_node"]
        self.kp = setting["dropout"]
        self.pred_term = setting["Fit"]
        if key != "":
            self.term_target = self.term_target[self.macro_seq_len + self.pred_term :]
            # print('===>', len(self.term_target))
        self.lr = setting["lr_rate"]
        self.epoch = setting["epoch"]
        self.batch_num = setting["Tbatch"]
        self.val_cut, self.te_cut = setting["val"], setting["te"]
        self.val_batch = setting["Vbatch"]
        self.shuffle = setting["shuffle"]
        self.save_dir = dir + "\\"
        param = pd.DataFrame()
        for i, (k, v) in enumerate(self.setting.items()):
            if type(v) == list:
                self.save_dir += "_{}{}".format(k.split("_")[0], len(v))
                param = pd.concat(
                    [
                        param,
                        pd.DataFrame(
                            v, index=[k + str(i) for i in range(len(v))], columns=[0]
                        ),
                    ]
                )
            else:
                if i == 0:
                    self.save_dir += "{}{}".format(k.split("_")[0], v)
                else:
                    self.save_dir += "_{}{}".format(k.split("_")[0], v)
                param = pd.concat([param, pd.DataFrame(v, index=[k], columns=[0])])
        os.makedirs(self.save_dir, exist_ok=True)
        param.to_csv(self.save_dir + "/param.csv")
        print(self.save_dir)

        self.predict = 23
        self.test_interval = int(np.ceil(self.predict / self.batch_num)) - 1
        # print(self.test_interval)

    def data_split(self, shuffle=False):
        """S = [0, 0, 1], M = [0, 1, 0], L = [1, 0, 0]"""
        target, input = pd.get_dummies(self.term_target["close_term"]).dropna(), []
        self.date = target.index
        data_info = {}
        for i in self.date:
            bw_23 = pd.date_range(end=i, freq="B", periods=self.pred_term)[0]
            input.append(
                minmax_scale(
                    np.array(
                        self.macro_data.loc[
                            pd.date_range(
                                end=bw_23, freq="B", periods=self.macro_seq_len
                            )[0] : bw_23
                        ]
                    )
                )
            )
            data_info.update(
                {
                    "target_date": i,
                    "input_start": pd.date_range(
                        end=bw_23, freq="B", periods=self.macro_seq_len
                    )[0],
                    "input_end": bw_23,
                }
            )

        input = np.array(input)
        target = np.array(target)

        if self.val_cut != 0 and self.te_cut != 0:
            te_num, val_num = (
                int(len(input) * self.te_cut),
                int(len(input) * self.val_cut),
            )

            while te_num % self.batch_num != 0:
                te_num -= 1
            while val_num % self.batch_num != 0:
                val_num -= 1
            self.te_x = input[-te_num:]
            self.te_y = target[-te_num:]
            self.val_x = input[-(val_num + te_num) : -te_num]
            self.val_y = target[-(val_num + te_num) : -te_num]
            self.tr_x = input[: -(val_num + te_num)]
            self.tr_y = target[: -(val_num + te_num)]
            self.tr_date = self.date[: -(val_num + te_num)]
            if shuffle == True:
                data_len = len(self.tr_x)
                run_shuffle = np.arange(data_len)
                np.random.shuffle(run_shuffle)
                self.tr_x = self.tr_x[run_shuffle]
                self.tr_y = self.tr_y[run_shuffle]
                self.tr_date = self.date[: -(val_num + te_num)][run_shuffle]
            print(
                "X tr_shape : {} , val_shape : {} , te_shape : {}".format(
                    self.tr_x.shape, self.val_x.shape, self.te_x.shape
                )
            )
            print(
                "Y tr_shape : {} , val_shape : {} , te_shape : {}".format(
                    self.tr_y.shape, self.val_y.shape, self.te_y.shape
                )
            )

            split_info = pd.DataFrame(
                {
                    "train_st": self.date[: -(val_num + te_num)][0],
                    "train_end": self.date[: -(val_num + te_num)][-1],
                    "val_st": self.date[-(val_num + te_num) : -te_num][0],
                    "val_end": self.date[-(val_num + te_num) : -te_num][-1],
                    "test_st": self.date[-te_num:][0],
                    "test_end": self.date[-te_num:][-1],
                },
                index=[0],
            )
            split_info.to_csv(self.save_dir + "/data_split.csv")
            self.te_st, self.te_end = (
                split_info.loc[0, "test_st"],
                split_info.loc[0, "test_end"],
            )
            batch_index = sum(
                [
                    [
                        i
                        for i in range(int(np.ceil(len(self.tr_x) / self.batch_num)))
                        for _ in range(self.batch_num)
                    ][: len(self.tr_x)],
                    [
                        i
                        for i in range(int(len(self.val_x) / self.batch_num))
                        for _ in range(self.batch_num)
                    ],
                    [
                        i
                        for i in range(int(len(self.te_x) / self.batch_num))
                        for _ in range(self.batch_num)
                    ],
                ],
                [],
            )
            batch_date = sum(
                [
                    [i for i in self.tr_date],
                    [i for i in self.date[-(val_num + te_num) : -te_num]],
                    [i for i in self.date[-te_num:]],
                ],
                [],
            )
            input_end = [
                pd.date_range(end=i, freq="B", periods=self.pred_term)[0]
                for i in batch_date
            ]
            input_start = [
                pd.date_range(end=i, freq="B", periods=self.macro_seq_len)[0]
                for i in input_end
            ]
            batch_info = pd.DataFrame(
                {
                    "target_date": batch_date,
                    "input_start": input_start,
                    "input_end": input_end,
                },
                index=batch_index,
            )
            batch_info.to_csv(self.save_dir + "/batch_info.csv")

        elif self.val_cut == 0 and self.te_cut != 0:
            te_num = int(len(input) * self.te_cut)
            while te_num % self.batch_num != 0:
                te_num -= 1
            self.tr_x = input[:-te_num]
            self.tr_y = target[:-te_num]
            self.tr_date = self.date[:-te_num]
            self.te_x = input[-te_num:]
            self.te_y = target[-te_num:]
            self.te_date = self.date[-te_num:]

            if shuffle == True:
                data_len = len(self.tr_x)
                run_shuffle = np.arange(data_len)
                np.random.shuffle(run_shuffle)
                self.tr_x = self.tr_x[run_shuffle]
                self.tr_y = self.tr_y[run_shuffle]
                self.tr_date = self.tr_date[run_shuffle]

            batch_index = sum(
                [
                    [
                        i
                        for i in range(int(np.ceil(len(self.tr_x) / self.batch_num)))
                        for _ in range(self.batch_num)
                    ][: len(self.tr_x)],
                    [
                        i
                        for i in range(int(len(self.te_x) / self.batch_num))
                        for _ in range(self.batch_num)
                    ],
                ],
                [],
            )
            batch_date = sum(
                [[i for i in self.tr_date], [i for i in self.date[-te_num:]]], []
            )
            input_end = [
                pd.date_range(end=i, freq="B", periods=self.pred_term)[0]
                for i in batch_date
            ]
            input_start = [
                pd.date_range(end=i, freq="B", periods=self.macro_seq_len)[0]
                for i in input_end
            ]
            batch_info = pd.DataFrame(
                {
                    "target_date": batch_date,
                    "input_start": input_start,
                    "input_end": input_end,
                },
                index=batch_index,
            )
            batch_info.to_csv(self.save_dir + "/batch_info.csv")

        else:
            self.tr_x = input
            self.tr_y = target
            batch_index = [
                i
                for i in range(int(np.ceil(len(self.tr_x) / self.batch_num)))
                for _ in range(self.batch_num)
            ][: len(self.tr_x)]
            input_end = [
                pd.date_range(end=i, freq="B", periods=self.pred_term)[0]
                for i in self.date
            ]
            input_start = [
                pd.date_range(end=i, freq="B", periods=self.macro_seq_len)[0]
                for i in input_end
            ]
            batch_info = pd.DataFrame(
                {
                    "target_date": self.date,
                    "input_start": input_start,
                    "input_end": input_end,
                },
                index=batch_index,
            )
            batch_info.to_csv(self.save_dir + "/batch_info.csv")

    def train(self):

        self.data_split(self.shuffle)
        self.phase = tf.placeholder(tf.bool, name="phase")
        self.keep_prob = tf.placeholder(tf.float32)
        self.X = tf.placeholder(
            tf.float32, shape=[None, self.macro_seq_len, self.macro_feature]
        )
        self.Y = tf.placeholder(tf.float32, shape=[None, 3])

        self.model = self.LSTM()
        l2_loss = tf.losses.get_regularization_loss()
        self.cost = (
            tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.Y, logits=self.model
                )
            )
            + l2_loss
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.cost
        )
        self.acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1)), tf.float32
            )
        )

    def split_running(self):

        self.train()
        ## GPU setting
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.InteractiveSession(
            config=config
        )  # tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        ## Learning
        saver = tf.train.Saver(max_to_keep=200)
        checkpoint = tf.train.get_checkpoint_state(
            self.save_dir + "/model_save/lstm_model"
        )  # _batch{}'.format(self.batch_size))
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Succesfully loaded:", checkpoint.model_checkpoint_path)
        else:
            learning_info = pd.DataFrame()
            test_set = pd.DataFrame()

            for i in range(self.epoch):

                tr_cost, val_cost = 0, 0
                tr_acc, val_acc, te_acc = 0, 0, 0

                ## TRAIN
                iteration = int(np.ceil(len(self.tr_x) / self.batch_num))
                for batch in range(0, len(self.tr_x), self.batch_num):
                    X, y = (
                        self.tr_x[batch : batch + self.batch_num],
                        self.tr_y[batch : batch + self.batch_num],
                    )
                    _, cost_val, acc_val = sess.run(
                        [self.optimizer, self.cost, self.acc],
                        feed_dict={
                            self.X: X,
                            self.Y: y,
                            self.phase: True,
                            self.keep_prob: self.kp,
                        },
                    )
                    tr_cost += cost_val / iteration
                    tr_acc += acc_val / iteration
                tr_check = sess.run(
                    tf.argmax(self.model, 1),
                    feed_dict={
                        self.X: self.tr_x,
                        self.Y: self.tr_y,
                        self.phase: True,
                        self.keep_prob: self.kp,
                    },
                )

                if i < 20:
                    print("== Weight_saving ==")
                    saver.save(
                        sess, self.save_dir + "/model_save/lstm_model", global_step=i
                    )

                ## VALIDATION
                iteration = int(np.ceil(len(self.val_x) / self.batch_num))
                for batch in range(0, len(self.val_x), self.batch_num):
                    X, y = (
                        self.val_x[batch : batch + self.batch_num],
                        self.val_y[batch : batch + self.batch_num],
                    )
                    cost_val, acc_val = sess.run(
                        [self.cost, self.acc],
                        feed_dict={
                            self.X: X,
                            self.Y: y,
                            self.phase: False,
                            self.keep_prob: 1.0,
                        },
                    )
                    val_cost += cost_val / iteration
                    val_acc += acc_val / iteration

                ## TEST
                iteration = int(np.ceil(len(self.te_x) / self.batch_num))
                for batch in range(0, len(self.te_x), self.batch_num):
                    X, y = (
                        self.te_x[batch : batch + self.batch_num],
                        self.te_y[batch : batch + self.batch_num],
                    )
                    acc_val = sess.run(
                        self.acc,
                        feed_dict={
                            self.X: X,
                            self.Y: y,
                            self.phase: False,
                            self.keep_prob: 1.0,
                        },
                    )
                    te_acc += acc_val / iteration
                te_check = sess.run(
                    tf.argmax(self.model, 1),
                    feed_dict={
                        self.X: self.te_x,
                        self.Y: self.te_y,
                        self.phase: False,
                        self.keep_prob: 1.0,
                    },
                )
                learning_info = pd.concat(
                    [
                        learning_info,
                        pd.DataFrame(
                            {
                                "tr_COST": round(tr_cost, 4),
                                "val_COST": round(val_cost, 4),
                                "tr_ACC": round(tr_acc, 4),
                                "val_ACC": round(val_acc, 4),
                                "te_ACC": round(te_acc, 4),
                                "tr_S": round(
                                    len(tr_check[tr_check == 2]) / len(tr_check), 4
                                ),
                                "tr_M": round(
                                    len(tr_check[tr_check == 1]) / len(tr_check), 4
                                ),
                                "tr_L": round(
                                    len(tr_check[tr_check == 0]) / len(tr_check), 4
                                ),
                                "te_S": round(
                                    len(te_check[te_check == 2]) / len(te_check), 4
                                ),
                                "te_M": round(
                                    len(te_check[te_check == 1]) / len(te_check), 4
                                ),
                                "te_L": round(
                                    len(te_check[te_check == 0]) / len(te_check), 4
                                ),
                            },
                            index=[i],
                        ),
                    ]
                )
                print(learning_info.loc[i])
                learning_info.index.name = "epoch"
                learning_info.to_csv(self.save_dir + "/learning_info.csv")

                te_check = pd.DataFrame(te_check)
                te_check[te_check == 2] = "S"
                te_check[te_check == 1] = "M"
                te_check[te_check == 0] = "L"
                te_check.index = pd.date_range(self.te_st, self.te_end, freq="B")
                te_check.columns = [i]
                test_set = pd.concat([test_set, te_check], axis=1)
                test_set.to_csv(self.save_dir + "/pred_term.csv")

    def split_online_running(self, training_online=False):

        self.train()
        ## GPU setting
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.InteractiveSession(
            config=config
        )  # tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        ## Saving
        saver = tf.train.Saver(max_to_keep=200)
        checkpoint = tf.train.get_checkpoint_state(
            self.save_dir + "/model_save/online_lstm"
        )  # _batch{}'.format(self.batch_size))
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Succesfully loaded:", checkpoint.model_checkpoint_path)
        else:

            ## Learning
            learning_info = pd.DataFrame()
            batch_set = [i for i in range(0, len(self.tr_x), self.batch_num)]
            # date = [self.tr_date[i] for i in range(0, len(self.tr_x)+1, self.batch_num)]
            print("training DATE : ", self.tr_date)

            if training_online == False:
                iteration = len(batch_set) - 1
                for i in range(self.epoch):
                    tr_cost = 0
                    for num in range(len(batch_set)):
                        trX, trY = (
                            self.tr_x[batch_set[num] : batch_set[num] + self.batch_num],
                            self.tr_y[batch_set[num] : batch_set[num] + self.batch_num],
                        )
                        _, cost_val = sess.run(
                            [self.optimizer, self.cost],
                            feed_dict={
                                self.X: trX,
                                self.Y: trY,
                                self.phase: True,
                                self.keep_prob: self.kp,
                            },
                        )
                        tr_cost += cost_val / iteration
                    tr_check = sess.run(
                        tf.argmax(self.model, 1),
                        feed_dict={
                            self.X: self.tr_x,
                            self.Y: self.tr_y,
                            self.phase: True,
                            self.keep_prob: self.kp,
                        },
                    )
                    learning_info = pd.concat(
                        [
                            learning_info,
                            pd.DataFrame(
                                {
                                    "tr_COST": round(tr_cost, 4),
                                    "tr_S": round(
                                        len(tr_check[tr_check == 2]) / len(tr_check), 4
                                    ),
                                    "tr_M": round(
                                        len(tr_check[tr_check == 1]) / len(tr_check), 4
                                    ),
                                    "tr_L": round(
                                        len(tr_check[tr_check == 0]) / len(tr_check), 4
                                    ),
                                },
                                index=[i],
                            ),
                        ]
                    )
                    print(learning_info.iloc[-1:])
                    learning_info.index.name = "train_start"
                    learning_info.to_csv(self.save_dir + "/learning_info.csv")
                # print("== Weight_saving ==")
                # saver.save(sess, self.save_dir + '/model_save/lstm_training')

            ## training set online
            elif training_online == True:
                for num in range(len(batch_set)):
                    trX, trY = (
                        self.tr_x[batch_set[num] : batch_set[num] + self.batch_num],
                        self.tr_y[batch_set[num] : batch_set[num] + self.batch_num],
                    )
                    tr_cost, tr_acc = 0, 0

                    for i in range(self.epoch):
                        ## TRAIN
                        _, cost_val, acc_val = sess.run(
                            [self.optimizer, self.cost, self.acc],
                            feed_dict={
                                self.X: trX,
                                self.Y: trY,
                                self.phase: True,
                                self.keep_prob: self.kp,
                            },
                        )
                        tr_cost += cost_val / self.epoch
                        tr_acc += acc_val / self.epoch
                        if i == self.epoch - 1:
                            tr_check = sess.run(
                                tf.argmax(self.model, 1),
                                feed_dict={
                                    self.X: self.tr_x,
                                    self.Y: self.tr_y,
                                    self.phase: True,
                                    self.keep_prob: self.kp,
                                },
                            )

                    learning_info = pd.concat(
                        [
                            learning_info,
                            pd.DataFrame(
                                {
                                    "tr_COST": round(tr_cost, 4),
                                    "tr_ACC": round(tr_acc, 4),
                                    "tr_S": round(
                                        len(tr_check[tr_check == 2]) / len(tr_check), 4
                                    ),
                                    "tr_M": round(
                                        len(tr_check[tr_check == 1]) / len(tr_check), 4
                                    ),
                                    "tr_L": round(
                                        len(tr_check[tr_check == 0]) / len(tr_check), 4
                                    ),
                                },
                                index=[num],
                            ),
                        ]
                    )
                    print(learning_info.iloc[-1:])
                    learning_info.index.name = "batch_num"
                    learning_info.to_csv(self.save_dir + "/learning_info.csv")

                # print("== Weight_saving ==")
                # saver.save(sess, self.save_dir + '/model_save/lstm_learning_tr')

            ## validation online learning
            if self.val_cut != 0:

                batch_set = [i for i in range(0, len(self.val_x), self.val_batch)]
                val_date = self.date[
                    -(len(self.val_x) + len(self.te_x)) : -len(self.te_x)
                ]
                print("validation DATE : ", val_date)
                date = [val_date[i] for i in range(0, len(self.val_x), self.val_batch)]
                self.val_check = pd.DataFrame()
                learning_info = pd.DataFrame()
                data_len = np.arange(self.batch_num)

                for num in range(1, len(batch_set)):
                    X, y = (
                        self.val_x[
                            batch_set[num - 1] : batch_set[num - 1] + self.batch_num
                        ],
                        self.val_y[
                            batch_set[num - 1] : batch_set[num - 1] + self.batch_num
                        ],
                    )

                    for batch in range(0, self.val_batch, self.batch_num):
                        if self.shuffle == True:
                            np.random.shuffle(data_len)
                        for i in range(self.epoch):
                            sess.run(
                                self.optimizer,
                                feed_dict={
                                    self.X: X[batch : batch + self.batch_num][data_len],
                                    self.Y: y[batch : batch + self.batch_num][data_len],
                                    self.phase: True,
                                    self.keep_prob: self.kp,
                                },
                            )
                    # [sess.run(self.optimizer, feed_dict={self.X: X, self.Y: y, self.phase: True, self.keep_prob: self.kp}) for _ in range(self.epoch)]
                    val_cost = sess.run(
                        self.cost,
                        feed_dict={
                            self.X: X,
                            self.Y: y,
                            self.phase: True,
                            self.keep_prob: 1.0,
                        },
                    )
                    teX, tey = (
                        self.val_x[batch_set[num] : batch_set[num] + self.batch_num],
                        self.val_y[batch_set[num] : batch_set[num] + self.batch_num],
                    )
                    val_acc = sess.run(
                        self.acc,
                        feed_dict={
                            self.X: teX,
                            self.Y: tey,
                            self.phase: True,
                            self.keep_prob: 1.0,
                        },
                    )
                    val_check = sess.run(
                        tf.argmax(self.model, 1),
                        feed_dict={
                            self.X: teX,
                            self.Y: tey,
                            self.phase: True,
                            self.keep_prob: self.kp,
                        },
                    )

                    learning_info = pd.concat(
                        [
                            learning_info,
                            pd.DataFrame(
                                {
                                    "val_COST": round(val_cost, 4),
                                    "val_ACC": round(val_acc, 4),
                                    "val_S": round(
                                        len(val_check[val_check == 2]) / len(val_check),
                                        4,
                                    ),
                                    "val_M": round(
                                        len(val_check[val_check == 1]) / len(val_check),
                                        4,
                                    ),
                                    "val_L": round(
                                        len(val_check[val_check == 0]) / len(val_check),
                                        4,
                                    ),
                                },
                                index=[date[num]],
                            ),
                        ]
                    )
                    learning_info.index.name = "train_start"
                    learning_info.to_csv(self.save_dir + "/validation_info.csv")
                    print(learning_info.iloc[-1:])

                # print("== Weight_saving ==")
                # saver.save(sess, self.save_dir + '/model_save/lstm_learning_val')

                self.te_check = pd.DataFrame()
                te_date = self.date[-len(self.te_x) :]
                print("test DATE : ", te_date)
                self.te_x = self.te_x[self.predict :]
                date = [te_date[i] for i in range(0, len(self.te_x), self.batch_num)]
                batch_set = [i for i in range(0, len(self.te_x), self.batch_num)]
                learning_info = pd.DataFrame()

                for num in range(len(batch_set)):
                    X, y = (
                        self.te_x[batch_set[num] : batch_set[num] + self.batch_num],
                        self.te_y[batch_set[num] : batch_set[num] + self.batch_num],
                    )
                    acc_val = sess.run(
                        self.acc,
                        feed_dict={
                            self.X: X,
                            self.Y: y,
                            self.phase: True,
                            self.keep_prob: 1.0,
                        },
                    )
                    new = pd.DataFrame(
                        sess.run(
                            tf.argmax(self.model, 1),
                            feed_dict={
                                self.X: X,
                                self.Y: y,
                                self.phase: True,
                                self.keep_prob: 1.0,
                            },
                        ),
                        index=te_date[batch_set[num] : batch_set[num] + self.batch_num],
                        columns=["pred"],
                    )
                    self.te_check = pd.concat([self.te_check, new])
                    learning_info = pd.concat(
                        [
                            learning_info,
                            pd.DataFrame(
                                {
                                    "te_ACC": round(acc_val, 4),
                                    "te_S": round((new["pred"] == 2).mean(), 4),
                                    "te_M": round((new["pred"] == 1).mean(), 4),
                                    "te_L": round((new["pred"] == 0).mean(), 4),
                                },
                                index=[date[num]],
                            ),
                        ]
                    )
                    learning_info.index.name = "test_start"
                    learning_info.to_csv(self.save_dir + "/test_info.csv")
                    print(learning_info.iloc[-1:])

                self.te_check[self.te_check == 2] = "S"
                self.te_check[self.te_check == 1] = "M"
                self.te_check[self.te_check == 0] = "L"
                self.te_check.to_csv(self.save_dir + "/PRED_TERM.csv")
                print(learning_info["te_ACC"].mean())

            ## No validation set / test online
            else:

                self.te_check = pd.DataFrame()
                date = [
                    self.te_date[i] for i in range(0, len(self.te_x), self.batch_num)
                ]
                batch_set = [i for i in range(0, len(self.te_x), self.batch_num)]
                learning_info = pd.DataFrame()
                data_len = np.arange(self.batch_num)

                for num in range(1, len(batch_set) - self.test_interval):
                    X, y = (
                        self.te_x[
                            batch_set[num - 1] : batch_set[num - 1] + self.batch_num
                        ],
                        self.te_y[
                            batch_set[num - 1] : batch_set[num - 1] + self.batch_num
                        ],
                    )
                    if self.shuffle == True:
                        np.random.shuffle(data_len)
                    [
                        sess.run(
                            self.optimizer,
                            feed_dict={
                                self.X: X[data_len],
                                self.Y: y[data_len],
                                self.phase: True,
                                self.keep_prob: self.kp,
                            },
                        )
                        for _ in range(self.epoch)
                    ]
                    teX, teY = (
                        self.te_x[
                            batch_set[num + self.test_interval] : batch_set[
                                num + self.test_interval
                            ]
                            + self.batch_num
                        ],
                        self.te_y[
                            batch_set[num + self.test_interval] : batch_set[
                                num + self.test_interval
                            ]
                            + self.batch_num
                        ],
                    )
                    print(
                        "training -- {} / test -- {}".format(
                            num - 1, num + self.test_interval
                        )
                    )
                    acc_val = sess.run(
                        self.acc,
                        feed_dict={
                            self.X: teX,
                            self.Y: teY,
                            self.phase: True,
                            self.keep_prob: 1.0,
                        },
                    )
                    new = pd.DataFrame(
                        sess.run(
                            tf.argmax(self.model, 1),
                            feed_dict={
                                self.X: X,
                                self.Y: y,
                                self.phase: True,
                                self.keep_prob: 1.0,
                            },
                        ),
                        index=self.te_date[
                            batch_set[num + self.test_interval] : batch_set[
                                num + self.test_interval
                            ]
                            + self.batch_num
                        ],
                        columns=["pred"],
                    )
                    self.te_check = pd.concat([self.te_check, new])
                    learning_info = pd.concat(
                        [
                            learning_info,
                            pd.DataFrame(
                                {
                                    "te_ACC": round(acc_val, 4),
                                    "te_S": round((new["pred"] == 2).mean(), 4),
                                    "te_M": round((new["pred"] == 1).mean(), 4),
                                    "te_L": round((new["pred"] == 0).mean(), 4),
                                },
                                index=[date[num + self.test_interval]],
                            ),
                        ]
                    )
                    learning_info.to_csv(self.save_dir + "/test_info.csv")
                self.te_check[self.te_check == 2] = "S"
                self.te_check[self.te_check == 1] = "M"
                self.te_check[self.te_check == 0] = "L"
                self.te_check.to_csv(self.save_dir + "/PRED_TERM.csv")
                print(learning_info["te_ACC"].mean())

    def online_learning(self):

        self.train()
        ## GPU setting
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.InteractiveSession(
            config=config
        )  # tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        ## Learning
        learning_info = pd.DataFrame()
        self.te_check = pd.DataFrame()
        date = [self.date[i] for i in range(0, len(self.tr_x), self.batch_num)]
        batch_set = [i for i in range(0, len(self.tr_x), self.batch_num)]
        data_len = np.arange(self.batch_num)

        for num in range(len(batch_set) - 2 - self.test_interval):
            if self.shuffle == True:
                np.random.shuffle(data_len)

            tr_cost = 0
            for i in range(self.epoch):
                trX, trY = (
                    self.tr_x[batch_set[num] : batch_set[num + 1]],
                    self.tr_y[batch_set[num] : batch_set[num + 1]],
                )
                _, cost_val = sess.run(
                    [self.optimizer, self.cost],
                    feed_dict={
                        self.X: trX[data_len],
                        self.Y: trY[data_len],
                        self.phase: True,
                        self.keep_prob: self.kp,
                    },
                )
                tr_cost += cost_val / self.epoch
            teX, teY = (
                self.tr_x[
                    batch_set[num + 1 + self.test_interval] : batch_set[
                        num + 2 + self.test_interval
                    ]
                ],
                self.tr_y[
                    batch_set[num + 1 + self.test_interval] : batch_set[
                        num + 2 + self.test_interval
                    ]
                ],
            )
            print(
                "training -- {} / test -- {}".format(num, num + 1 + self.test_interval)
            )
            acc_val = sess.run(
                self.acc,
                feed_dict={
                    self.X: teX,
                    self.Y: teY,
                    self.phase: True,
                    self.keep_prob: 1.0,
                },
            )
            new = pd.DataFrame(
                sess.run(
                    tf.argmax(self.model, 1),
                    feed_dict={
                        self.X: teX,
                        self.Y: teY,
                        self.phase: True,
                        self.keep_prob: 1.0,
                    },
                ),
                index=self.date[
                    batch_set[num + 1 + self.test_interval] : batch_set[
                        num + 2 + self.test_interval
                    ]
                ],
                columns=["pred"],
            )
            self.te_check = pd.concat([self.te_check, new])
            learning_info = pd.concat(
                [
                    learning_info,
                    pd.DataFrame(
                        {
                            "tr_Cost": round(tr_cost, 4),
                            "te_ACC": round(acc_val, 4),
                            "te_S": round((new["pred"] == 2).mean(), 4),
                            "te_M": round((new["pred"] == 1).mean(), 4),
                            "te_L": round((new["pred"] == 0).mean(), 4),
                        },
                        index=[date[num + 1 + self.test_interval]],
                    ),
                ]
            )
            learning_info.index.name = "test_start"
            learning_info.to_csv(self.save_dir + "/learning_info.csv")
        self.te_check[self.te_check == 2] = "S"
        self.te_check[self.te_check == 1] = "M"
        self.te_check[self.te_check == 0] = "L"
        self.te_check.to_csv(self.save_dir + "/PRED_TERM.csv")
        print(learning_info["te_ACC"].mean())

    def run(self):

        if self.learning == "split":
            self.split_running()
        elif self.learning == "split_online":
            self.split_online_running(True)
        elif self.learning == "online":
            self.online_learning()
