from config.Loader import Default

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU
from functools import partial
import pickle


class Predictor(tf.keras.Model, Default):

    def __init__(self):
        super(Predictor, self).__init__(name='Predictor')
        Default.__init__(self)

        name2layer = {
            "Dense": partial(Dense, activation=self.dense_activation, dtype='float32'),
            "LSTM" : partial(LSTM, time_major=False, dtype='float32', stateful=False, return_sequences=True),
            "GRU"  : partial(GRU, time_major=False, dtype='float32', stateful=False, return_sequences=True)
        }

        self.model_layers = [
            name2layer[name](dim, name="%s_%d" % (name, i)) for i, (name, dim) in enumerate(self.model)
        ]

        self.BG = Dense(1, activation='linear', dtype='float32', name="BG")

    def call(self, data):
        for layer in self.model_layers:
            data = layer(data)

        return self.BG(data)[:, :, 0]

    def get_params(self):
        return [layer.get_weights() for layer in self.model_layers]

    def set_params(self, params):
        for param, layer in zip(params,  self.model_layers):
            layer.set_weights(param)

    def save_checkpoint(self, path=None, name=""):
        if path is None:
            path = self.ckpt_path
        with open(path+name+".pkl", 'wb+') as f:
            pickle.dump(self.get_params(), f)

    def load_checkpoint(self, path=None, name=""):
        if path is None:
            path = self.ckpt_path
        with open(path+name+'.pkl', 'rb') as f:
            self.set_params(pickle.load(f))


class Trainer(tf.keras.Model):
    def __init__(self, param_dict):
        super(Trainer, self).__init__(name='trainer')

        self.model = Predictor()
        self.params = param_dict
        #self.optim = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-8, clipvalue=4e-3)
        self.optim = tf.keras.optimizers.Adam(learning_rate=param_dict['learning_rate'])
        self.step = 0

    def train(self, data, gpu=0):
        # do some stuff with arrays
        # print(states, actions, rewards, dones)
        # Set both networks with corresponding initial recurrent state

        grad_norm, loss, error_rate = self._train(data['input'], data['to_predict'], int(gpu))
        self.step += 1

        tf.summary.scalar(name="train/loss", data=loss)
        tf.summary.scalar(name="train/grad_norm", data=grad_norm)
        tf.summary.scalar(name="train/error_rate", data=error_rate)

        return loss, error_rate

    @tf.function
    def _train(self, input_data, to_predict, gpu):
        '''
        Main training function
        '''

        with tf.device("/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"):

            with tf.GradientTape() as tape:

                predicted = self.model(input_data)

                loss = tf.reduce_mean(tf.square(predicted-to_predict))

        grad = tape.gradient(loss, self.model.trainable_variables)

        # x is used to track the gradient size
        x = 0.0
        c = 0.0
        for gg in grad:
            c += 1.0
            x += tf.reduce_mean(tf.abs(gg))
        x /= c

        error_rate = self.error_rate(predicted, to_predict)

        self.optim.apply_gradients(zip(grad, self.model.trainable_variables))

        return x, loss, error_rate

    def test(self, data):
        return self._test(data['input'], data['to_predict'])

    @tf.function
    def _test(self, input_data, to_predict, gpu=0):
        with tf.device("/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"):
            predicted = self.model(input_data)

        return self.error_rate(predicted, to_predict), predicted

    @staticmethod
    def error_rate(predicted, to_predict):
        return tf.reduce_mean(tf.abs(predicted-to_predict)/(to_predict+1e-8))


