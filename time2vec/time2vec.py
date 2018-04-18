#!/usr/bin/env python

"""
| File name: time2vec.py
| Author: Francois Meyer
| Email: francoisrmeyer@gmail.com
| Supervisors: Brink van der Merwe (Stellenbosch University), Dirko Coetsee (Praelexis)
| Date created: 2017/07/03
| Date last modified: 2018/04/17
| Python Version: 3.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import math
import csv
import gc
import logging
import pickle
from multiprocessing import Process, Lock
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

logger = logging.getLogger()
logging.basicConfig(format='%(message)s', level=logging.INFO)


class Time2Vec(object):
    """
    Class that contains a model for learning concept embeddings from temporal data. It can be used to set up a model by
    specifying various parameters, generate training data from a specified data set for the model, train concept
    embeddings from the training data and investigate the properties of obtained embeddings.
    """

    def __init__(self, data_file_name='data.csv', min_count=1, subsampling=0.001, train_file_name='training_data.csv',
                 decay=0, unit=0, const=10., rate=3., limit=None, chunk_size=100000, processes=4, dimen=100,
                 num_samples=20, optimizer=0, lr=0.025, min_lr=None, batch_size=32, epochs=5, valid=0.0, seed=1):
        """
        Initialise the model by specifying a data set from which to learn the embeddings, as well as a number of
        parameters that specify the details of the model and how embeddings are learned. The numerical parameters are
        verified to be in their expected ranges.

        :param data_file_name: name or path of the CSV file containing the data set used to learn embeddings, with each
            line containing the data (id, datetime, concept) for an event
        :param min_count: minimum number of occurrences of a concept required, otherwise the concept is ignored
        :param subsampling: relative frequency threshold for determining how concepts are downsampled

        :param train_file_name: name or path of the CSV file to which is generated training data is written or from
            which existing training data is read
        :param decay: type of decay with which event influence decays over time (0=no decay, 1=linear, 2=exponential)
            linear formula: influence = const - rate * time_difference_between_events,
            exponential formula: influence = const * exp(-rate * time_difference_between_events)
        :param unit: unit of time used to determine decay of event influence (0=year, 1=month, 2=day)
        :param const: constant weight from which influence decays linearly
        :param rate: rate at which the influence of events should decay with time
        :param limit: time interval after which event influence completely cuts off
        :param chunk_size: number of records to be read into memory during training data generation
        :param processes: number of worker processes to use during training data generation

        :param dimen: dimensionality of the concept embeddings
        :param num_samples: number of negative samples to be drawn
        :param optimizer: stochastic optimization technique used during training (0=gradient descent, 1=Adam, 2=Adagrad)
        :param lr: initial learning rate used during training
        :param min_lr: learning rate will drop linearly to this during training (only for the case of stochastic
            gradient descent)
        :param batch_size: number of training samples given to the learning algorithm per training iteration
        :param epochs: number of training iterations over the entire training data set
        :param valid: proportion of the training data to be used for validation
        :param seed: random seed used during the initialisation of values in the neural network (for reproducibility)
        """

        # Parameters for building the vocabulary
        self.data_file_name = data_file_name

        if min_count < 1:
            raise ValueError('The parameter min_count must be a positive integer.')
        self.min_count = min_count

        if subsampling < 0.0 or subsampling > 1.0:
            raise ValueError('The parameter subsampling must be a float in the range [0, 1].')
        self.subsampling = subsampling

        # Parameters for generating training data
        self.train_file_name = train_file_name

        if unit != 0 and unit != 1 and unit != 2:
            raise ValueError('The parameter unit must be 0 (year), 1 (month) or 2 (day).')
        self.unit = unit

        if const <= 0.:
            raise ValueError('The parameter const must be a positive float.')
        self.const = const

        if rate < 0.:
            raise ValueError('The parameter rate must be a non-negative float.')
        self.rate = rate

        if decay != 0 and decay != 1 and decay != 2:
            raise ValueError('The parameter decay must be 0 (no decay), 1 (linear decay) or 2 (exponential decay).')
        self.decay = decay

        if limit is not None:
            if limit <= 0:
                raise ValueError('The parameter limit must be a positive integer.')
        self.limit = limit

        if chunk_size <= 0:
            raise ValueError('The parameter chunk_size must be a positive integer.')
        self.chunk_size = chunk_size

        if processes <= 0:
            raise ValueError('The parameter processes must be a positive integer.')
        self.processes = processes

        # Parameters for learning embeddings
        if dimen <= 0:
            raise ValueError('The parameter dimen must be a positive integer.')
        self.dimen = dimen

        if num_samples <= 0:
            raise ValueError('The parameter num_samples must be a positive integer.')
        self.num_samples = num_samples

        if optimizer != 0 and optimizer != 1 and optimizer != 2:
            raise ValueError('The parameter optimizer must be 0 (gradient descent), 1 (Adam aglorithm) or 2 '
                             '(adagrad algorithm).')
        self.optimizer = optimizer

        if lr <= 0.:
            raise ValueError('The parameter lr must be a positive float.')
        self.lr = lr

        if min_lr is not None:
            if min_lr <= 0. or min_lr >= lr:
                raise ValueError('If the parameter min_lr is set, it must be a positive float and less than lr.')
        self.min_lr = min_lr

        if batch_size <= 0:
            raise ValueError('The parameter batch_size must be a positive integer.')
        self.batch_size = batch_size

        if num_samples <= 0:
            raise ValueError('The parameter epochs must be a positive integer.')
        self.epochs = epochs

        if valid < 0.0 or valid > 1.0:
            raise ValueError('The parameter valid must be a float in the range [0, 1].')
        self.valid = valid

        self.seed = seed

    def train(self):
        """
        Build the concept vocabulary, generate and store training data and learn embeddings from the training data.
        All of this is done with the model specified by the current parameter values. After training the concept
        embeddings are available.
        """

        # Gather and store required vocabulary information from data
        self.build_vocab()

        # Transform data to be used for training
        self.gen_train_data()

        # Learn embeddings from training data
        self.learn_embeddings()

    def build_vocab(self):
        """
        Gather and store required information about the concept 'vocabulary'. This includes counting concept occurrences
        in order to implement the minimum count requirement and subsampling, as well as constructing a lookup table that
        maps all concepts in the data set to unique indices.
        """

        logger.info('Building vocabulary...')

        iterator = VocabIterator(self.data_file_name)
        vocab = {}

        # Count concept occurrences
        raw_vocab_count = {}
        raw_vocab = []
        total_records = 0
        for line in iterator:
            concept = line[2]
            total_records += 1
            if concept not in raw_vocab_count:
                raw_vocab_count[concept] = 1
                raw_vocab.append(concept)
            else:
                raw_vocab_count[concept] += 1

        # Set up vocabulary
        vocab_count = {}
        data_count = 0
        for concept in raw_vocab:
            count = raw_vocab_count[concept]
            if count >= self.min_count:
                index = len(vocab)
                vocab[concept] = index
                vocab_count[index] = count
                if self.subsampling != 0:
                    data_count += count

        raw_vocab_count.clear()
        del raw_vocab[:]

        # Store all required information
        self.total_records = total_records
        self.vocab = vocab
        self.vocab_count = vocab_count
        self.vocab_size = len(vocab)
        self.reverse_vocab = {index: concept for concept, index in vocab.items()}
        self.vocab_series = pd.Series(vocab)
        self.data_count = data_count
        if self.subsampling != 0:
            self.threshold_count = self.subsampling * self.data_count

        logger.info('Vocabulary complete.')

    def gen_train_data(self):
        """
        Iterate through the data set, transforming it to the format required for training and storing the generated
        training data. The training data should consist of a sequence of input-target training pairs with associated
        weights. The input-target pairs should consist of the indices of concepts and the associated weights should be
        the influence assigned to the training pairs, determined by the time interval between the concept occurrences.
        """

        logger.info('Generating training data...')
        start_time = datetime.now()

        iterator = DataIterator(self.data_file_name, self.chunk_size)

        # Open and empty file to store training data in
        train_file = open(self.train_file_name, 'w')
        train_file.close()

        # Define lock to protect write access to training data file
        lock = Lock()

        # Job offloaded to processes
        def job(proc_data):
            proc_train_file = open(self.train_file_name, 'a')
            proc_train_data = self.get_train_data(proc_data)
            with lock:
                proc_train_data.to_csv(proc_train_file, header=False, index=False)

        # Set up multiprocessing
        procs = []
        count = 0
        num_processes = self.processes

        # Iterate through all events in the data set
        records_processed = 0
        next_data = None
        done = False
        while True:
            gc.collect()

            # Read new chunk of records to process
            new_data = iterator.next()
            if new_data is None or len(new_data.index) < self.chunk_size:
                done = True
            if next_data is None:
                curr_data = new_data
            else:
                curr_data = next_data.append(new_data)

            # No new data and no next data
            if curr_data is None:
                break

            curr_data = curr_data.reset_index(drop=True)
            curr_data['datetime'] = pd.to_datetime(curr_data['datetime'])

            # Store records for last ID
            if not done:
                last_id = curr_data.loc[len(curr_data) - 1][0]
                next_data = curr_data[curr_data.id == last_id]
                curr_data = curr_data[curr_data.id != last_id]

            # Transform the data to training data and store in CSV file
            if not curr_data.empty:

                if num_processes > 1:
                    # All processes full, wait for all to finish
                    if count % num_processes == 0 and count != 0:
                        for p in procs:
                            p.join()
                        progress = records_processed / float(self.total_records) * 100
                        curr_time = datetime.now()
                        delta = curr_time - start_time
                        elapsed = delta.total_seconds()
                        logger.info('Processed ' + str(records_processed) + ' records : ' + str(round(progress, 2))
                                    + '% in ' + str(elapsed) + ' seconds.')
                        count = 0

                    # Assign data to next available process
                    if len(procs) == num_processes:
                        procs[count] = Process(target=job, args=(curr_data,))
                    else:
                        procs.append(Process(target=job, args=(curr_data,)))
                    procs[count].start()
                    count += 1

                    records_processed += len(curr_data.index)

                else:
                    # Perform operations in single process
                    curr_train_data = self.get_train_data(curr_data)
                    train_file = open(self.train_file_name, 'a')
                    curr_train_data.to_csv(train_file, header=False, index=False)
                    train_file.close()
                    records_processed += len(curr_data.index)
                    progress = records_processed / float(self.total_records) * 100
                    curr_time = datetime.now()
                    delta = curr_time - start_time
                    elapsed = delta.total_seconds()
                    logger.info('Processed ' + str(records_processed) + ' records : ' + str(round(progress, 2))
                                + '% in ' + str(elapsed) + ' seconds.')

            if done:
                # Wait for all processes to finish
                if num_processes > 1:
                    for p in procs[:count]:
                        p.join()
                    if count > 0:
                        progress = records_processed / float(self.total_records) * 100
                        logger.info('Processed ' + str(records_processed) + ' records : ' + str(round(progress, 2)) + '%')
                break

        train_file.close()

        # Load training data
        self.load_train_data()

        end_time = datetime.now()
        delta = end_time - start_time
        elapsed = delta.total_seconds()

        logger.info('Training data generated in ' + str(elapsed) + ' seconds.')

    def get_train_data(self, data):
        """
        Transform the raw data to the format required for training. This consists of transforming a sequence of events
        to a set of weighted input-target training samples. This is the data format required by the neural network to
        learn concept embeddings.

        :param data: data frame with columns [id, datetime, concept], containing a sequence of events
        :return: training_data: data frame with columns [input, target, weight], containing the training samples and
            the weights to be associated with them
        """

        # Replace the concepts in the data set with their indices
        data_df = data.copy()
        data_df['concept'] = data_df['concept'].map(self.vocab_series)

        # Discard words that do not occur enough
        if self.min_count > 1:
            data_df = data_df.dropna().reset_index()

        # Subsample frequently occurring concepts
        if self.subsampling > 0:

            def get_probability(row):
                # Calculate the probability of retaining a row
                index = row['concept']
                p = (math.sqrt(self.vocab_count[index] / self.threshold_count) + 1) *\
                    (self.threshold_count / self.vocab_count[index])
                p = min(p, 1.0)
                return p

            # Keep concepts according to probabilities (may give numerical errors later!)
            data_df['prob'] = data_df.apply(lambda row: get_probability(row), axis=1)
            rand = np.random.uniform(0, 1, len(data_df))
            data_df['rand'] = rand
            data_df = data_df.loc[data_df['rand'] < data_df['prob']]

        # Set up data frame for training data
        columns = ['input', 'target', 'weight']
        train_data = pd.DataFrame(columns=columns)
        train_data['input'] = train_data['input'].astype(np.int32)
        train_data['target'] = train_data['target'].astype(np.int32)
        train_data['weight'] = train_data['weight'].astype(np.float32)

        # Transform set events to input-target training pairs with training weights
        for center_index, center_row in data_df.iterrows():
            curr_id = center_row['id']

            # Look forward in time
            curr_index = center_index + 1
            while curr_index < len(data_df.index) and data_df.iloc[curr_index]['id'] == curr_id:
                context_row = data_df.iloc[curr_index]

                weight = self.get_weight(center_row['datetime'], context_row['datetime'])
                curr_index += 1

                # Stop looking
                if weight == 0:
                    break

                # Predict forward
                forward_row = pd.DataFrame(columns=columns,
                                           data=[[center_row['concept'], context_row['concept'], weight]])
                train_data = train_data.append(forward_row, ignore_index=True)

                # Predict backward
                backward_row = pd.DataFrame(columns=columns,
                                            data=[[context_row['concept'], center_row['concept'], weight]])
                train_data = train_data.append(backward_row, ignore_index=True)

        return train_data

    def load_train_data(self, training_file_name=None):
        """
        Load existing training data from a file, instead of generating it.

        :param training_file_name: name or path of the CSV file from which existing training data is read (if
            unspecified, the data is read from the training file specified during model initialisation)
        """

        if training_file_name is not None:
            self.train_file_name = training_file_name

        self.train_data = TrainIterator(self.train_file_name, self.batch_size)

        self.data_size = self.train_data.count()
        self.valid_size = int(self.valid * self.data_size)
        self.train_size = self.data_size - self.valid_size

        batches_per_epoch = int(math.ceil(float(self.train_size) / self.batch_size))
        self.steps = self.epochs * batches_per_epoch

    def get_weight(self, date1, date2):
        """
        Calculate and return the weight to be associated with a training sample based on the datetimes of the two events
        of the training sample and the specified temporal parameters.

        :param date1: datetime of first event
        :param date2: datetime of second event
        :return: weight: training weight to be associated with sample
        """

        # Calculate difference in units
        if self.unit == 0:      # year
            diff_years = date1.year - date2.year
            diff = abs(diff_years)

        elif self.unit == 1:    # month
            diff_years = date1.year - date2.year
            diff_months = date1.month - date2.month
            diff = abs(diff_years * 12 + diff_months)

        else:                   # day
            diff_dates = date1 - date2
            diff = abs(diff_dates.days)

        # Check if difference is more than limit
        if self.limit is not None and diff > self.limit:
            return 0

        # Calculate weight based on decay
        if self.decay == 0:     # none
            return 1

        elif self.decay == 1:     # linear
            weight = self.const - self.rate * diff

        else:                   # exponential
            weight = self.const * np.exp(- self.rate * diff)

        weight = max(weight, 0)
        return weight

    def set_vocab_params(self, data_file_name=None, min_count=None, subsampling=None):
        """
        Set the vocabulary parameters. If these parameters are changed, the entire training procedure must be restarted
        - build_vocab and gen_train_data must be called before embeddings can be learned.
        """

        if data_file_name is not None:
            self.data_file_name = data_file_name

        if min_count is not None:
            if min_count < 1:
                raise ValueError('The parameter min_count must be a positive integer.')
            self.min_count = min_count

        if subsampling is not None:
            if subsampling < 0.0 or subsampling > 1.0:
                raise ValueError('The parameter subsampling must be a float in the range [0, 1].')
            self.subsampling = subsampling

        logging.warning('Because of the parameter changes the entire training procedure must be restarted build_vocab'+
                        'and gen_train_data must be called before embeddings can be learned with the new parameters.')

    def set_train_data_params(self, train_file_name=None, decay=None, unit=None, const=None, rate=None, limit=None,
                              chunk_size=None, processes=None):
        """
        Set the training data parameters. If these parameters are changed, the training data must be regenerated -
        gen_train_data must be called again before embeddings can be learned.
        """

        if train_file_name is not None:
            self.train_file_name = train_file_name

        if unit is not None:
            if unit != 0 and unit != 1 and unit != 2:
                raise ValueError('The parameter unit must be 0 (year), 1 (month) or 2 (day).')
            self.unit = unit

        if const is not None:
            if const <= 0.:
                raise ValueError('The parameter const must be a positive float.')
            self.const = const

        if rate is not None:
            if rate < 0.:
                raise ValueError('The parameter rate must be a non-negative float.')
            self.rate = rate

        if decay is not None:
            if decay != 0 and decay != 1 and decay != 2:
                raise ValueError('The parameter decay must be 0 (no decay), 1 (linear decay) or 2 (exponential decay).')
            self.decay = decay

        if limit is not None:
            if limit <= 0:
                raise ValueError('The parameter limit must be a positive integer.')
        self.limit = limit

        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError('The parameter chunk_size must be a positive integer.')
            self.chunk_size = chunk_size

        if processes is not None:
            if processes <= 0:
                raise ValueError('The parameter processes must be a positive integer.')
            self.processes = processes

        logging.warning('Because of the parameter changes the training data must be regenerated - gen_train_data must'+
                        'be called again before embeddings can be learned with the new parameters.')

    def set_learn_params(self, dimen=None, num_samples=None, optimizer=None, lr=None, min_lr=None, batch_size=None,
                         epochs=None, valid=None, seed=None):
        """
        Set the learning parameters. If these parameters are changed, it is not required to call any other functions
        again before embeddings can be learned.
        """

        if dimen is not None:
            if dimen <= 0:
                raise ValueError('The parameter dimen must be a positive integer.')
            self.dimen = dimen

        if num_samples is not None:
            if num_samples <= 0:
                raise ValueError('The parameter num_samples must be a positive integer.')
            self.num_samples = num_samples

        if optimizer is not None:
            if optimizer != 0 and optimizer != 1 and optimizer != 2:
                raise ValueError('The parameter optimizer must be 0 (gradient descent), 1 (Adam aglorithm) or 2 '
                                 '(adagrad algorithm).')
            self.optimizer = optimizer

        if lr is not None:
            if lr <= 0.:
                raise ValueError('The parameter lr must be a positive float.')
            self.lr = lr

        if min_lr is not None:
            if min_lr <= 0. or min_lr >= lr:
                raise ValueError('If the parameter min_lr is set, it must be a positive float and less than lr.')
        self.min_lr = min_lr

        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError('The parameter batch_size must be a positive integer.')
            self.batch_size = batch_size

        if num_samples is not None:
            if num_samples <= 0:
                raise ValueError('The parameter epochs must be a positive integer.')
            self.epochs = epochs

        if valid is not None:
            if valid < 0.0 or valid > 1.0:
                raise ValueError('The parameter valid must be a float in the range [0, 1].')
            self.valid = valid

        if seed is not None:
            self.seed = seed

    def learn_embeddings(self):
        """
        Learns concept embeddings from the training data with a neural network set up similarly to those used to obtain
        word embeddings. The neural network is implemented in TensorFlow.
        Construction phase: set up the neural network according the the specified learning parameters.
        Execution phase: feed training batches to the neural network and learn concept embeddings.
        """

        logger.info('Training model...')
        start_time = datetime.now()

        # Define placeholders for the training data
        with tf.name_scope('placeholders'):
            inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
            weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            targets = tf.placeholder(tf.int32, shape=[None,1], name='targets')

        # Define weights for the hidden layer (embedding matrix) and lookup input embeddings
        tf.set_random_seed(self.seed)
        init_width = 0.5 / self.dimen
        init_embeddings = tf.random_uniform([self.vocab_size, self.dimen], -init_width, init_width, seed=self.seed)
        with tf.name_scope('input_embeddings'):
            embeddings = tf.Variable(init_embeddings, name='embeddings')
            input_embeddings = tf.nn.embedding_lookup(embeddings, inputs, name='input_embeddings')

        # Define weights and biases for the output layer
        init_weights = tf.zeros([self.vocab_size, self.dimen])
        init_biases = tf.zeros([self.vocab_size])
        with tf.name_scope('output_embeddings'):
            output_weights = tf.Variable(init_weights, name='output_weights')
            output_biases = tf.Variable(init_biases, name='output_biases')

        # Define sampling distribution for negative samples
        vocab_count_list = [self.vocab_count[key] for key in sorted(self.vocab_count.keys())]
        with tf.name_scope('noise_samples'):
            sampled_candidates, true_expected_count, sampled_expected_count = \
                tf.nn.fixed_unigram_candidate_sampler(true_classes=tf.cast(targets, tf.int64),
                                                      num_true=1,
                                                      num_sampled=self.num_samples,
                                                      unique=True,
                                                      range_max=self.vocab_size,
                                                      distortion=0.75,
                                                      unigrams=vocab_count_list)

        # Define the NCE loss and compute the cost function with weighted losses
        with tf.name_scope('loss'):
            losses = tf.nn.nce_loss(weights=output_weights,
                                    biases=output_biases,
                                    labels=targets,
                                    inputs=input_embeddings,
                                    num_sampled=self.num_samples,
                                    num_classes=self.vocab_size,
                                    sampled_values=(sampled_candidates, true_expected_count, sampled_expected_count),
                                    name='losses')

            weighted_loss = tf.losses.compute_weighted_loss(losses=losses, weights=weights)

        # Define the training step
        with tf.name_scope('optimizer'):
            if self.optimizer == 0:
                # Gradient descent
                if self.min_lr is not None:
                    # Linear learning rate decay
                    global_step = tf.Variable(0, trainable=False)
                    decaying_lr = tf.train.polynomial_decay(learning_rate=self.lr, global_step=global_step,
                                                            decay_steps=self.steps, end_learning_rate=self.min_lr)
                    train = tf.train.GradientDescentOptimizer(decaying_lr).minimize(weighted_loss,
                                                                                    global_step=global_step,
                                                                                    var_list=[embeddings,
                                                                                              output_weights,
                                                                                              output_biases],
                                                                                    name='train')
                else:
                    # No learning rate decay
                    train = tf.train.GradientDescentOptimizer(self.lr).minimize(weighted_loss,
                                                                                var_list=[embeddings, output_weights,
                                                                                          output_biases],
                                                                                name='train')

            elif self.optimizer == 1:
                # Adam algorithm
                train = tf.train.AdamOptimizer(self.lr).minimize(weighted_loss,
                                                                 var_list=[embeddings, output_weights, output_biases],
                                                                 name='train')

            else:
                # Adagrad algorithm
                train = tf.train.AdagradOptimizer(self.lr).minimize(weighted_loss,
                                                                    var_list=[embeddings, output_weights,
                                                                              output_biases],
                                                                    name='train')

        # Train the model
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for i in range(self.epochs):

                if i % math.ceil(self.epochs/10.) == 0:
                    progress = i / self.epochs * 100
                    logger.info('Training progress at ' + str(progress) + '%')

                # New epoch of training and validation
                training_phase = True
                count = 0
                valid_loss = 0.0

                while True:
                    batch_data = self.train_data.next()

                    if batch_data is None:
                        # Reached end of epoch
                        self.train_data.restart()
                        break

                    curr_batch_size = len(batch_data)
                    count += curr_batch_size

                    if training_phase:

                        if count > self.train_size:
                            # Reached end of training data
                            training_phase = False

                            curr_valid_size = count - self.train_size
                            curr_train_size = curr_batch_size - curr_valid_size

                            # Remaining training data
                            batch_inputs = batch_data[:curr_train_size, 0]
                            batch_targets = batch_data[:curr_train_size, 1].reshape((len(batch_inputs), 1))
                            batch_weights = batch_data[:curr_train_size, 2]

                            sess.run(train, {inputs: batch_inputs, targets: batch_targets, weights: batch_weights})

                            # Start with validation data
                            batch_inputs = batch_data[curr_train_size:, 0]
                            batch_targets = batch_data[curr_train_size:, 1].reshape((len(batch_inputs), 1))
                            batch_weights = batch_data[curr_train_size:, 2]

                            curr_valid_loss = sess.run(weighted_loss, {inputs: batch_inputs, targets: batch_targets,
                                                                       weights: batch_weights})
                            valid_loss += curr_valid_loss

                        else:
                            # Continue with training data
                            batch_inputs = batch_data[:, 0]
                            batch_targets = batch_data[:, 1].reshape((len(batch_inputs), 1))
                            batch_weights = batch_data[:, 2]

                            sess.run([train], {inputs: batch_inputs, targets: batch_targets, weights: batch_weights})

                    else:
                        # Continue with validation data
                        batch_inputs = batch_data[:, 0]
                        batch_targets = batch_data[:, 1].reshape((len(batch_inputs), 1))
                        batch_weights = batch_data[:, 2]

                        curr_valid_loss = sess.run(weighted_loss, {inputs: batch_inputs, targets: batch_targets,
                                                                   weights: batch_weights})
                        valid_loss += curr_valid_loss

                if self.valid_size > 0.0:
                    logger.info("Epoch " + str(i+1) + " validation loss: " + str(valid_loss))

            self.final_embeddings = embeddings.eval()

        logger.info('Training progress at 100%')

        end_time = datetime.now()
        delta = end_time - start_time
        elapsed = delta.total_seconds()

        logger.info('Model trained in ' + str(elapsed) + ' seconds.')

    def get_vector(self, concept_name):
        """
        Returns the vector representation of a specific concept, as learned by the model.

        :param concept_name: concept for which the vector representation is found
        :return: concept_vector: numpy array containing the vector for the concept
        """

        if concept_name not in self.vocab:
            raise ValueError('The concept ' + concept_name + ' is not in the model vocabulary.')

        # Lookup concept index and concept vector
        concept_index = self.vocab[concept_name]
        concept_vector = self.final_embeddings[concept_index]
        return concept_vector

    def most_similar(self, concept_name, k=1):
        """
        Find and return the k most similar concepts to a given concept with regards to cosine similarities of the
        vectors of the concepts.

        :param concept_name: concept for which the most similar concepts are found
        :param k: number of most similar concepts to find and return
        :return: nearest: list of tuples containing the names and cosine similarities of the k most similar concepts,
            ranked by descending cosine similarity
        """

        if k is not None:
            if k <= 0:
                raise ValueError('The parameter k must be a positive integer.')

        # Calculate the cosine similarities of the concept vector and all other concepts' vectors
        concept_vector = self.get_vector(concept_name)
        similarities = cosine_similarity([concept_vector], self.final_embeddings)

        # Rank indices according to descending cosine similarities and find most similar concepts
        nearest_indices = (-similarities).argsort()[:self.vocab_size]
        nearest_indices = nearest_indices[0,0:k+1]
        nearest_concepts = [self.reverse_vocab[index] for index in nearest_indices]
        nearest_similars = [similarities[0,index] for index in nearest_indices]

        # Remove concept itself
        concept_index = nearest_concepts.index(concept_name)
        nearest_concepts.remove(concept_name)
        del nearest_similars[concept_index]

        nearest_concepts = nearest_concepts[0:k]
        nearest_similars = nearest_similars[0:k]
        nearest = list(zip(nearest_concepts, nearest_similars))

        return nearest

    def save(self, save_file_name):
        """
        Save the embeddings learned by the current model to disk.

        :param save_file_name: name or path of the file used to store the embeddings
        """

        save_file = open(save_file_name, 'wb', pickle.HIGHEST_PROTOCOL)
        model_embeddings = {'vocab': self.vocab, 'vocab_size': self.vocab_size, 'reverse_vocab': self.reverse_vocab,
                            'final_embeddings': self.final_embeddings}
        pickle.dump(model_embeddings, save_file)


def load(load_file_name):
    """
    Load learned embeddings from disk.

    :param load_file_name: name or path of the file from which the embeddings are loaded
    :return model: an instance of Time2Vec with the loaded embeddings
    """

    load_file = open(load_file_name, 'rb')
    model_embeddings = pickle.load(load_file)

    # Create an instance of Time2Vec and restore the embeddings to it
    model = Time2Vec()
    model.vocab = model_embeddings['vocab']
    model.vocab_size = model_embeddings['vocab_size']
    model.reverse_vocab = model_embeddings['reverse_vocab']
    model.final_embeddings = model_embeddings['final_embeddings']
    return model


class VocabIterator(object):
    """
    Wrapper class that can be used to iterate through the original data set, stored in a CSV file, while building the
    vocabulary.
    """

    def __init__(self, file_name):
        """
        Initialise an instance of the class by creating an iterator from the specified file.

        :param file_name: name of the CSV file from which the iterator reads data
        """
        self.file = open(file_name)
        self.iterator = csv.reader(self.file)

    def __iter__(self):
        for line in self.iterator:
            yield line


class DataIterator(object):
    """
    Wrapper class that can be used to iterate through the original data set, stored in a CSV file, while generating the
    training data.
    """

    def __init__(self, file_name, chunk_size):
        """
        Initialise an instance of the class by creating an iterator from the specified file.

        :param file_name: name of the CSV file from which the iterator reads data
        """
        self.file = open(file_name)
        columns = ['id', 'datetime', 'concept']
        self.reader = pd.read_csv(filepath_or_buffer=self.file, header=None, names=columns, iterator=True,
                                  chunksize=chunk_size)

    def __iter__(self):
        return self.reader

    def next(self):
        """
        :return: record: next record in the data set being iterated
        """
        data = next(self.reader, None)
        return data


class TrainIterator(object):
    """
    Wrapper class that can be used to iterate through the training data set, stored in a CSV file, to obtain training
    sample batches during training.
    """

    def __init__(self, file_name, batch_size):
        """
        Initialise an instance of the class by creating an iterator from the specified file.

        :param file_name: name of the CSV file from which the iterator reads data
        """
        self.batch_size = batch_size
        self.file_name = file_name
        self.reader = pd.read_csv(filepath_or_buffer=file_name, header=None, iterator=True, chunksize=batch_size)

    def __iter__(self):
        return self.reader

    def next(self):
        """
        :return: record: next record in the data set being iterated
        """
        data = next(self.reader, None)

        if data is not None:
            data_array = data.values
        else:
            data_array = None
        return data_array

    def restart(self):
        """
        Restart the iterator. Iteration will resume at the start of the training data set.
        """
        self.reader = pd.read_csv(filepath_or_buffer=self.file_name, header=None, iterator=True,
                                  chunksize=self.batch_size)

    def count(self):
        """
        Count the number of training samples
        :return:
        """
        # Calculate total training steps
        train_samples_count = 0
        chunk_size = 100000

        while True:
            try:
                chunk = self.reader.get_chunk(size=chunk_size)
                size = len(chunk.index)
                if size < chunk_size:
                    train_samples_count += len(chunk.index)
                else:
                    train_samples_count += chunk_size
            except StopIteration:
                break

        self.restart()
        return train_samples_count

# Example usage with a toy medical data set.
if __name__ == '__main__':

    medical_data_file = '../example_data/medical.csv'
    medical_training_file = '../example_data/medical_training_data.csv'

    # Initialise the model
    model = Time2Vec(data_file_name=medical_data_file,
                     min_count=1,
                     subsampling=0,
                     train_file_name=medical_training_file,
                     decay=1,
                     unit=2,
                     const=1.,
                     rate=0.3,
                     limit=5,
                     chunk_size=1000,
                     processes=4,
                     dimen=100,
                     num_samples=5,
                     optimizer=2,
                     lr=1.0,
                     min_lr=0.1,
                     batch_size=16,
                     epochs=1000,
                     valid=0,
                     seed=1)

    # Training process
    model.build_vocab()
    model.gen_train_data()
    model.learn_embeddings()

    # Explore embeddings
    closest = model.most_similar('disease1', 4)
    print(closest)
