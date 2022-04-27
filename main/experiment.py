from config.Loader import Default
from logger.logger import Logger
from .nn import Trainer, Predictor
from .stats import Stats
from .data_processing import DataBag


import matplotlib.pyplot as plt
plt.style.use(['science'])
import os
import fire
import tensorflow as tf
import datetime
import numpy as np


class Experiment(Default, Logger):

    def __init__(self):
        super(Experiment, self).__init__()

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


        self.running_instance_id = datetime.datetime.now().strftime("BG_TIMESERIES_%Y-%m-%d_%H-%M")
        log_dir = 'logs/' + self.running_instance_id
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()
        tf.summary.experimental.set_step(0)

        self.learning_params = {
            'learning_rate'    : self.learning_rate,
            'n_epoch'          : self.n_epoch,
            'batch_size'       : self.batch_size,
            'trajectory_length': self.trajectory_length,
        }

        self.data = None
        self.train_stats = Stats("Loss", "Error rate")
        self.test_stats = Stats("Error rate")
        self.models = [Predictor() for _ in range(12)]
        self.trainer = Trainer(self.learning_params)

    def general_train(self):
        self.logger.info('Training model on training data...')
        for epoch in range(self.n_epoch):
            for i, training_data in enumerate(self.data.training_data()):
                loss, error_rate = self.trainer.train(training_data, self.gpu)
                tf.summary.experimental.set_step(self.trainer.step)
                self.train_stats['Loss'] = loss
                self.train_stats['Error rate'] = error_rate
                if i % self.log_freq == 0:
                    self.logger.info(
                        'Epoch: %s | %.2f%% | Loss: %.6f | Error rate: %.3f' % (str(1 + epoch).rjust(2, ' '),
                                                                                100 * i / len(
                                                                                    self.data.data['training']),
                                                                                self.train_stats['Loss'],
                                                                                self.train_stats['Error rate']))

        self.logger.info('Testing model on testing data...')
        for i, training_data in enumerate(self.data.testing_data()):
            self.test_stats['Error rate'], predicted = self.trainer.test(training_data)
            if i % self.log_freq == 0:
                self.logger.info('Error rate: %.3f' % (self.test_stats['Error rate']))

    def fine_tuning(self):

        base_weights = self.trainer.model.get_weights()
        for ind_num, model in enumerate(self.models):
            self.trainer.set_weights(base_weights)

            self.logger.info('Training model %d on for individual %d ...' % (ind_num, ind_num))
            for epoch in range(self.n_epoch):
                for i, training_data in enumerate(self.data.training_data(ind_num)):
                    loss, error_rate = self.trainer.train(training_data, self.gpu)
                    tf.summary.experimental.set_step(self.trainer.step)
                    self.train_stats['Loss'] = loss
                    self.train_stats['Error rate'] = error_rate
                    if i % self.log_freq == 0:
                        self.logger.info(
                            'Epoch: %s | %.2f%% | Loss: %.6f | Error rate: %.3f' % (str(1 + epoch).rjust(2, ' '),
                                                                                    100 * i / len(
                                                                                        self.data.data['training']),
                                                                                    self.train_stats['Loss'],
                                                                                    self.train_stats['Error rate']))

            self.logger.info('Testing model on testing data...')
            for i, training_data in enumerate(self.data.testing_data(ind_num)):
                self.test_stats['Error rate'], predicted = self.trainer.test(training_data)
                if i % self.log_freq == 0:
                    self.logger.info('Error rate: %.3f' % (self.test_stats['Error rate']))

            #model.set_weights(self.trainer.get_weights())
            sample = np.concatenate(np.concatenate(self.data.data_individual['testing'][ind_num], axis=0), axis=0)
            print(sample)
            to_predict = sample[self.n_step_predict:, self.data.to_predict_index]

            #normalized = (sample[np.newaxis, :-self.n_step_predict] - self.data.means) / self.data.std
            # predicted = self.trainer.predict_n_step(normalized, look_aheads)
            predicted = self.trainer.model(sample[np.newaxis, :-self.n_step_predict])[0]

            plt.figure(figsize=(7,5), dpi=300)

            plt.plot(np.arange(len(to_predict)), to_predict * self.data.std[self.data.to_predict_index] + self.data.means[
                         self.data.to_predict_index], label="Real")
            plt.plot(np.arange(len(to_predict)),
                     predicted * self.data.std[self.data.to_predict_index] + self.data.means[
                         self.data.to_predict_index],
                     label="%d min prediction" % (self.n_step_predict * 5))
            plt.xlabel('Time')
            plt.ylabel('Blood glucose')
            plt.legend()
            plt.savefig(self.result_path+str(ind_num+1)+'.png')
            self.trainer.model.save_checkpoint(name=str(ind_num+1))
            plt.clf()


    def __call__(self):
        self.logger.info('Experiment started.')
        try:
            self.logger.info('Loading data...')
            self.data = DataBag(self.batch_size, self.trajectory_length, self.n_step_predict)

            self.logger.info('Done.')

            # Use all the data to first train the model
            self.general_train()

            # For each individual, fine tune the model using data specific to that individual
            self.fine_tuning()

        except KeyboardInterrupt:

            pass

if __name__ == '__main__':
    fire.Fire(Experiment)




