# Time2Vec #

Time2Vec learns embeddings for concepts based on the occurrences of the concepts in a temporal data set. This is done by extending Word2Vec's Skip-gram architecture to take the time interval between events into account.

The model used to learn the embeddings can be specified by various parameters that relate to the way that the time at which events occur in the data set influence the way that the model learns the embeddings.

### Dependencies (Python) ###

* Numpy
* Pandas
* TensorFlow
* Scikit-Learn

### Source Data ###

The data set from which the embeddings are learned must consist of a sequence events of corresponding to occurrences of the concepts. The events must be in the format [id, datetime, concept]:

* id refers to an identifier with which each event is associated (this is used to group related events together)
* datetime specifies the date and time at which the event occurred
* concept specifies the concept that occurred

### Example Usage ###

```python
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
```
