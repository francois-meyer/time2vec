from unittest import TestCase
import pandas as pd
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, '../time2vec')
import time2vec


class TestTime2Vec(TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up test data
        cls.data_file_name = 'test_data.csv'
        cls.train_file_name = 'test_training_data.csv'
        cls.expected_train_data = np.array(
            [[0, 1, 10.], [1, 0, 10.], [0, 2, 10.], [2, 0, 10.], [1, 2, 10.], [2, 1, 10.],
             [0, 1, 10.], [1, 0, 10.], [0, 3, 7.], [3, 0, 7.], [1, 3, 7.], [3, 1, 7.],
             [0, 1, 10.], [1, 0, 10.], [0, 4, 4.], [4, 0, 4.], [1, 4, 4.], [4, 1, 4.]])

    def setUp(self):
        self.model = time2vec.Time2Vec(data_file_name=self.data_file_name,
                                       min_count=1,
                                       subsampling=0,
                                       train_file_name=self.train_file_name,
                                       decay=1,
                                       unit=2,
                                       const=10.,
                                       rate=3.,
                                       limit=None,
                                       chunk_size=1000,
                                       processes=4,
                                       dimen=100,
                                       num_samples=3,
                                       optimizer=0,
                                       lr=1.0,
                                       min_lr=0.1,
                                       batch_size=32,
                                       epochs=10,
                                       valid=0,
                                       seed=1)

    def test_train(self):
        self.model.train()

        self.assertEqual(self.model.final_embeddings.shape, (self.model.vocab_size, self.model.dimen))

    def test_build_vocab(self):
        self.model.build_vocab()
        expected_vocab = {'procedure':0, 'treatment':1, 'disease1':2, 'disease2':3, 'disease3':4}
        expected_reverse_vocab = {0:'procedure', 1:'treatment', 2:'disease1', 3:'disease2', 4:'disease3'}
        expected_vocab_count = {0:3, 1:3, 2:1, 3:1, 4:1}

        self.assertDictEqual(self.model.vocab, expected_vocab)
        self.assertEqual(self.model.vocab_size, len(expected_vocab))
        self.assertDictEqual(self.model.reverse_vocab, expected_reverse_vocab)
        pd.testing.assert_series_equal(self.model.vocab_series, pd.Series(self.model.vocab))
        self.assertDictEqual(self.model.vocab_count, expected_vocab_count)

    def test_build_vocab_min_count(self):
        self.model.set_vocab_params(min_count=2)
        self.model.build_vocab()
        expected_vocab = {'procedure':0, 'treatment':1}
        expected_reverse_vocab = {0:'procedure', 1:'treatment'}

        self.assertDictEqual(self.model.vocab, expected_vocab)
        self.assertEqual(self.model.vocab_size, len(expected_vocab))
        self.assertDictEqual(self.model.reverse_vocab, expected_reverse_vocab)
        pd.testing.assert_series_equal(self.model.vocab_series, pd.Series(self.model.vocab))

    def test_build_vocab_subsampling(self):
        self.model.set_vocab_params(subsampling=0.5)
        self.model.build_vocab()

        self.assertEqual(self.model.threshold_count, 4.5)

    def test_gen_train_data(self):
        self.model.build_vocab()
        self.model.gen_train_data()

        train_data = np.array(self.model.train_data.next())
        np.testing.assert_array_equal(train_data, self.expected_train_data)

    def test_load_train_data(self):
        self.model.load_train_data()

        train_data = np.array(self.model.train_data.next())
        np.testing.assert_array_equal(train_data, self.expected_train_data)

    def test_get_train_data(self):
        self.model.build_vocab()

        columns = ['id', 'datetime', 'concept']
        data_df = pd.read_csv(self.data_file_name, header=None, names=columns)
        data_df['datetime'] = pd.to_datetime(data_df['datetime'])
        train_data = self.model.get_train_data(data_df)

        np.testing.assert_array_equal(train_data, self.expected_train_data)

    def test_get_train_data_min_count(self):
        self.model.set_vocab_params(min_count=2)
        self.model.build_vocab()

        expected_train_data_min_count = np.array(
            [[0, 1, 10.], [1, 0, 10.], [0, 1, 10.], [1, 0, 10.], [0, 1, 10.], [1, 0, 10.]])

        columns = ['id', 'datetime', 'concept']
        data_df = pd.read_csv(self.data_file_name, header=None, names=columns)
        data_df['datetime'] = pd.to_datetime(data_df['datetime'])
        train_data = self.model.get_train_data(data_df)
        print(train_data)

        np.testing.assert_array_equal(train_data, expected_train_data_min_count)

    def test_get_train_data_decay_none(self):
        self.model.build_vocab()
        self.model.set_train_data_params(decay=0)
        expected_train_data = np.array(
            [[0, 1, 1.], [1, 0, 1.], [0, 2, 1.], [2, 0, 1.], [1, 2, 1.], [2, 1, 1.],
             [0, 1, 1.], [1, 0, 1.], [0, 3, 1.], [3, 0, 1.], [1, 3, 1.], [3, 1, 1.],
             [0, 1, 1.], [1, 0, 1.], [0, 4, 1.], [4, 0, 1.], [1, 4, 1.], [4, 1, 1.]])

        columns = ['id', 'datetime', 'concept']
        data_df = pd.read_csv(self.data_file_name, header=None, names=columns)
        data_df['datetime'] = pd.to_datetime(data_df['datetime'])
        train_data = self.model.get_train_data(data_df)

        np.testing.assert_array_equal(train_data, expected_train_data)

    def test_get_weight_unit_year(self):
        self.model.set_train_data_params(unit=0)

        date1 = datetime(2017, 6, 1, 12, 0, 0)
        date2 = datetime(2018, 6, 1, 12, 0, 0)
        date3 = datetime(2019, 6, 1, 12, 0, 0)
        date4 = datetime(2020, 6, 1, 12, 0, 0)
        date5 = datetime(2021, 6, 1, 12, 0, 0)

        weight1 = self.model.get_weight(date1, date1)
        weight2 = self.model.get_weight(date2, date1)
        weight3 = self.model.get_weight(date1, date3)
        weight4 = self.model.get_weight(date4, date1)
        weight5 = self.model.get_weight(date1, date5)

        self.assertEqual(weight1, 10.)
        self.assertEqual(weight2, 7.)
        self.assertEqual(weight3, 4.)
        self.assertEqual(weight4, 1.)
        self.assertEqual(weight5, 0.)

    def test_get_weight_unit_month(self):
        self.model.set_train_data_params(unit=1, const=20., rate=0.5)

        date1 = datetime(2017, 6, 1, 12, 0, 0)
        date2 = datetime(2017, 10, 1, 12, 0, 0)
        date3 = datetime(2018, 2, 1, 12, 0, 0)
        date4 = datetime(2020, 5, 1, 12, 0, 0)
        date5 = datetime(2030, 6, 1, 12, 0, 0)

        weight1 = self.model.get_weight(date1, date1)
        weight2 = self.model.get_weight(date2, date1)
        weight3 = self.model.get_weight(date1, date3)
        weight4 = self.model.get_weight(date4, date1)
        weight5 = self.model.get_weight(date1, date5)

        self.assertEqual(weight1, 20.)
        self.assertEqual(weight2, 18.)
        self.assertEqual(weight3, 16.)
        self.assertEqual(weight4, 2.5)
        self.assertEqual(weight5, 0.)

    def test_get_weight_unit_day(self):
        date1 = datetime(2017, 6, 1, 12, 0, 0)
        date2 = datetime(2017, 6, 2, 12, 0, 0)
        date3 = datetime(2017, 6, 3, 12, 0, 0)
        date4 = datetime(2017, 6, 4, 12, 0, 0)
        date5 = datetime(2017, 6, 5, 12, 0, 0)

        weight1 = self.model.get_weight(date1, date1)
        weight2 = self.model.get_weight(date2, date1)
        weight3 = self.model.get_weight(date1, date3)
        weight4 = self.model.get_weight(date4, date1)
        weight5 = self.model.get_weight(date1, date5)

        self.assertEqual(weight1, 10.)
        self.assertEqual(weight2, 7.)
        self.assertEqual(weight3, 4.)
        self.assertEqual(weight4, 1.)
        self.assertEqual(weight5, 0.)

    def test_get_weight_limit(self):
        self.model.set_train_data_params(limit=2)

        date1 = datetime(2017, 6, 1, 12, 0, 0)
        date2 = datetime(2017, 6, 2, 12, 0, 0)
        date3 = datetime(2017, 6, 3, 12, 0, 0)
        date4 = datetime(2017, 6, 4, 12, 0, 0)
        date5 = datetime(2017, 6, 5, 12, 0, 0)

        weight1 = self.model.get_weight(date1, date1)
        weight2 = self.model.get_weight(date2, date1)
        weight3 = self.model.get_weight(date1, date3)
        weight4 = self.model.get_weight(date4, date1)
        weight5 = self.model.get_weight(date1, date5)

        self.assertEqual(weight1, 10.)
        self.assertEqual(weight2, 7.)
        self.assertEqual(weight3, 4.)
        self.assertEqual(weight4, 0.)
        self.assertEqual(weight5, 0.)

    def test_get_weight_decay_exponential(self):
        self.model.set_train_data_params(decay=2, rate=1., limit=5)

        date1 = datetime(2017, 6, 1, 12, 0, 0)
        date2 = datetime(2017, 6, 2, 12, 0, 0)
        date3 = datetime(2017, 6, 3, 12, 0, 0)
        date4 = datetime(2017, 6, 4, 12, 0, 0)
        date5 = datetime(2017, 6, 5, 12, 0, 0)

        weight1 = self.model.get_weight(date1, date1)
        weight2 = self.model.get_weight(date2, date1)
        weight3 = self.model.get_weight(date1, date3)
        weight4 = self.model.get_weight(date4, date1)
        weight5 = self.model.get_weight(date1, date5)

        self.assertEqual(weight1, 10.)
        self.assertEqual(weight2, 3.6787944117144233)
        self.assertEqual(weight3, 1.353352832366127)
        self.assertEqual(weight4, 0.49787068367863946)
        self.assertEqual(weight5, 0.18315638888734179)

    def test_get_batch_data(self):
        self.model.set_learn_params(batch_size=5)

        expected_batch1 = np.array(self.expected_train_data[0:5, ])
        expected_batch2 = np.array(self.expected_train_data[5:10, ])
        expected_batch3 = np.array(self.expected_train_data[10:15, ])
        expected_batch4 = np.array(self.expected_train_data[15:, ])

        train_data = time2vec.TrainIterator(self.train_file_name, self.model.batch_size)
        batch1 = train_data.next()
        batch2 = train_data.next()
        batch3 = train_data.next()
        batch4 = train_data.next()

        np.testing.assert_array_equal(batch1, expected_batch1)
        np.testing.assert_array_equal(batch2, expected_batch2)
        np.testing.assert_array_equal(batch3, expected_batch3)
        np.testing.assert_array_equal(batch4, expected_batch4)

    def test_set_vocab_params(self):
        self.model.set_vocab_params(data_file_name='test_data2.csv', min_count=3, subsampling=0.1)

        self.assertEqual(self.model.data_file_name, 'test_data2.csv')
        self.assertEqual(self.model.min_count, 3)
        self.assertEqual(self.model.subsampling, 0.1)

    def test_set_train_data_params(self):
        self.model.set_train_data_params(train_file_name='test_training_data2.csv', decay=2, unit=2, const=3.,
                                         rate=0.25, limit=20, chunk_size=999, processes=16)

        self.assertEqual(self.model.train_file_name, 'test_training_data2.csv')
        self.assertEqual(self.model.decay, 2)
        self.assertEqual(self.model.unit, 2)
        self.assertEqual(self.model.const, 3.)
        self.assertEqual(self.model.rate, 0.25)
        self.assertEqual(self.model.limit, 20)
        self.assertEqual(self.model.chunk_size, 999)
        self.assertEqual(self.model.processes, 16)

    def test_set_learn_params(self):
        self.model.set_learn_params(dimen=300, num_samples=30, optimizer=2, lr=0.5, min_lr=0.05, batch_size=256,
                                    epochs=99, valid=0.1, seed=2)

        self.assertEqual(self.model.dimen, 300)
        self.assertEqual(self.model.num_samples, 30)
        self.assertEqual(self.model.optimizer, 2)
        self.assertEqual(self.model.lr, 0.5)
        self.assertEqual(self.model.min_lr, 0.05)
        self.assertEqual(self.model.batch_size, 256)
        self.assertEqual(self.model.epochs, 99)
        self.assertEqual(self.model.valid, 0.1)
        self.assertEqual(self.model.seed, 2)

    def test_learn_embeddings(self):
        self.model.build_vocab()
        self.model.gen_train_data()
        self.model.learn_embeddings()

        self.assertEqual(self.model.final_embeddings.shape, (self.model.vocab_size, self.model.dimen))

    def test_get_vector(self):
        self.model.build_vocab()
        self.model.gen_train_data()
        self.model.learn_embeddings()

        vector1 = self.model.get_vector('procedure')
        vector2 = self.model.get_vector('treatment')
        vector3 = self.model.get_vector('disease1')
        vector4 = self.model.get_vector('disease2')
        vector5 = self.model.get_vector('disease3')

        self.assertEqual(vector1.shape, (self.model.dimen,))
        self.assertEqual(vector2.shape, (self.model.dimen,))
        self.assertEqual(vector3.shape, (self.model.dimen,))
        self.assertEqual(vector4.shape, (self.model.dimen,))
        self.assertEqual(vector5.shape, (self.model.dimen,))

    def test_most_similar(self):
        self.model.build_vocab()
        self.model.gen_train_data()
        self.model.learn_embeddings()

        similars1 = self.model.most_similar('procedure')
        similars2 = self.model.most_similar('treatment', k=1)
        similars3 = self.model.most_similar('disease1', k=2)
        similars4 = self.model.most_similar('disease2', k=4)
        similars5 = self.model.most_similar('disease3', k=10)

        self.assertEqual(len(similars1), 1)
        self.assertEqual(len(similars2), 1)
        self.assertEqual(len(similars3), 2)
        self.assertEqual(len(similars4), 4)
        self.assertEqual(len(similars5), 4)
        self.assertEqual(len(similars1[0]), 2)
        self.assertEqual(len(similars3[1]), 2)
        self.assertEqual(len(similars5[3]), 2)