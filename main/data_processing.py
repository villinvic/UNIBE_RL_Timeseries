from config.Loader import Default

import pandas as pd
import os
import numpy as np


def split_dataframe(df, batch_size):
    chunks = list()
    num_chunks = len(df) // batch_size
    for i in range(num_chunks):
        chunks.append(df[len(df)-((i+1) * batch_size):len(df)-(i * batch_size)].to_numpy())
    return chunks

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


class DataBag(Default):
    to_predict_index = 0

    def __init__(self, batch_size, trajectory_length, n_step_predict=1):
        super(DataBag, self).__init__()

        self.batch_size = batch_size
        self.trajectory_length = trajectory_length + n_step_predict
        self.n_step_predict = n_step_predict

        self.pd_data = {
            'training': [],
            'testing': []
        }

        data_dim = self.load_data()
        self.scales = np.ones((data_dim,))

        self.data = {
            'training' : np.full((10000, self.batch_size, self.trajectory_length, data_dim), fill_value=np.nan, dtype=np.float32),
            'testing': np.full((10000, self.batch_size, self.trajectory_length, data_dim), fill_value=np.nan, dtype=np.float32),

        }

        self.data_individual = {
            'training': [np.full((1000, self.batch_size, self.trajectory_length, data_dim),
                                           fill_value=np.nan, dtype=np.float32) for _ in range(12)],
            'testing' : [np.full((1000, self.batch_size, self.trajectory_length, data_dim),
                                           fill_value=np.nan, dtype=np.float32) for _ in range(12)],
        }

        self.build_batches()



    def load_data(self):
        train_files = os.listdir(self.data_path + '2018/train') + os.listdir(self.data_path + '2020/train')
        test_files = os.listdir(self.data_path + '2018/test') + os.listdir(self.data_path + '2020/test')
        for path_string, files, data_bag in zip(['/train/', '/test/'], [train_files, test_files],
                                                self.pd_data.values()):
            for file in files:
                for d in ['2018', '2020']:
                    try :
                        with open(self.data_path + d + path_string + file, newline='') as fd:
                            data = pd.read_csv(fd, delimiter=',')
                    except:
                        pass
                data['cbg'] = data['cbg'].interpolate(method='linear', limit_direction='both')
                data['gsr'] = data['gsr'].interpolate(method='linear', limit_direction='both')
                data['finger'] = data['finger'].interpolate(method='linear', limit_direction='both')
                data['hr'] = data['hr'].fillna(method="ffill")
                data['hr'] = data['hr'].fillna(method="bfill")
                data['carbInput'] = data['carbInput'].fillna(0.)
                data['bolus'] = data['bolus'].fillna(0.)
                data['basal'] = data['basal'].interpolate(method='linear', limit_direction='both')

                data.fillna(0., inplace=True)

                data_bag.append(data)

        return len(self.pd_data['training'][0].columns) - 3 # get rid of timestamps and missing cbg

    def build_batches(self):
        for pd_data, (split_data_name, split_data) in zip(self.pd_data.values(), self.data.items()):
            index = 0
            for ind_num, data in enumerate(pd_data):
                index_ind = 0
                for trajectory in split_dataframe(data, self.trajectory_length):
                    if np.isnan(trajectory).any():
                        #print(data, index)
                        pass
                    split_data[(index // self.batch_size) % 10000, index % self.batch_size, :] = np.take(trajectory, axis=1, indices=[2,3,4,6,7,8]) # get rid of timestamps
                    self.data_individual[split_data_name][ind_num][(index_ind // self.batch_size) % 1000, index_ind % self.batch_size, :] = np.take(trajectory, axis=1, indices=[2,3,4,6,7,8])
                    index += 1
                    index_ind += 1
                self.data_individual[split_data_name][ind_num] = self.data_individual[split_data_name][ind_num][:(index_ind // self.batch_size) % 1000]
            self.data[split_data_name] = split_data[:(index // self.batch_size) % 10000]


        # normalize data
        all_data = np.concatenate([array for array in self.data.values()])
        self.means = np.nanmean(all_data, axis=(0,1,2))
        self.std = np.nanstd(all_data, axis=(0,1,2)) + 1e-8
        #maxes = np.maximum(*[np.nanmax(array, axis=(0,1,2)) for array in self.data.values()])
        #self.maxes = maxes + 1e-8
        for data in self.data.values():
            data[:] = (data[:] - self.means) / self.std
        for data in self.data_individual.values():
            for d in data:
                d[:] = (d[:] - self.means) / self.std


    def yield_data(self, datas, shuffle=True):
        if shuffle:
            x = np.take(datas, np.random.rand(datas.shape[0]).argsort(),axis=0)
            x = np.take(x, np.random.rand(x.shape[1]).argsort(), axis=1)
        else:
            x = datas

        for data in x:
            if not np.isnan(data).any():
                yield {
                    'input'     : data[:, :-self.n_step_predict],
                    'to_predict': data[:, self.n_step_predict:, self.to_predict_index]
                }
            else:
                pass


    def training_data(self, index=None):
        if index is None:
            for data in self.yield_data(self.data['training']):
                yield data
        else:
            for data in self.yield_data(self.data_individual['training'][index]):
                yield data


    def testing_data(self, index=None):
        if index is None:
            for data in self.yield_data(self.data['testing']):
                yield data
        else:
            for data in self.yield_data(self.data_individual['testing'][index]):
                yield data



# testing
if __name__ == '__main__':

    db = DataBag(64, 512)

    for full_data, to_predict in db.testing_data:
        print(to_predict.shape)
        print(full_data.shape)

