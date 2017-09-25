fasttext |Build Status| |PyPI version|
======================================

fasttext is a Python interface for `Facebook
fastText <https://github.com/facebookresearch/fastText>`__.

Requirements
------------

fasttext support Python 2.6 or newer. It requires
`Cython <https://pypi.python.org/pypi/Cython/>`__ in order to build the
C++ extension.

Installation
------------

.. code:: shell

    pip install fasttext

Example usage
-------------

This package has two main use cases: word representation learning and
text classification.

These were described in the two papers
`1 <#enriching-word-vectors-with-subword-information>`__ and
`2 <#bag-of-tricks-for-efficient-text-classification>`__.

Word representation learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to learn word vectors, as described in
`1 <#enriching-word-vectors-with-subword-information>`__, we can use
``fasttext.skipgram`` and ``fasttext.cbow`` function like the following:

.. code:: python

    import fasttext

    # Skipgram model
    model = fasttext.skipgram('data.txt', 'model')
    print model.words # list of words in dictionary

    # CBOW model
    model = fasttext.cbow('data.txt', 'model')
    print model.words # list of words in dictionary

where ``data.txt`` is a training file containing ``utf-8`` encoded text.
By default the word vectors will take into account character n-grams
from 3 to 6 characters.

At the end of optimization the program will save two files:
``model.bin`` and ``model.vec``.

``model.vec`` is a text file containing the word vectors, one per line.
``model.bin`` is a binary file containing the parameters of the model
along with the dictionary and all hyper parameters.

The binary file can be used later to compute word vectors or to restart
the optimization.

The following ``fasttext(1)`` command is equivalent

.. code:: shell

    # Skipgram model
    ./fasttext skipgram -input data.txt -output model

    # CBOW model
    ./fasttext cbow -input data.txt -output model

Obtaining word vectors for out-of-vocabulary words
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previously trained model can be used to compute word vectors for
out-of-vocabulary words.

.. code:: python

    print model['king'] # get the vector of the word 'king'

the following ``fasttext(1)`` command is equivalent:

.. code:: shell

    echo "king" | ./fasttext print-vectors model.bin

This will output the vector of word ``king`` to the standard output.

Load pre-trained model
~~~~~~~~~~~~~~~~~~~~~~

We can use ``fasttext.load_model`` to load pre-trained model:

.. code:: python

    model = fasttext.load_model('model.bin')
    print model.words # list of words in dictionary
    print model['king'] # get the vector of the word 'king'

Text classification
~~~~~~~~~~~~~~~~~~~

This package can also be used to train supervised text classifiers and
load pre-trained classifier from fastText.

In order to train a text classifier using the method described in
`2 <#bag-of-tricks-for-efficient-text-classification>`__, we can use the
following function:

.. code:: python

    classifier = fasttext.supervised('data.train.txt', 'model')

equivalent as ``fasttext(1)`` command:

.. code:: shell

    ./fasttext supervised -input data.train.txt -output model

where ``data.train.txt`` is a text file containing a training sentence
per line along with the labels. By default, we assume that labels are
words that are prefixed by the string ``__label__``.

We can specify the label prefix with the ``label_prefix`` param:

.. code:: python

    classifier = fasttext.supervised('data.train.txt', 'model', label_prefix='__label__')

equivalent as ``fasttext(1)`` command:

.. code:: shell

    ./fasttext supervised -input data.train.txt -output model -label '__label__'

This will output two files: ``model.bin`` and ``model.vec``.

Once the model was trained, we can evaluate it by computing the
precision at 1 (P@1) and the recall on a test set using
``classifier.test`` function:

.. code:: python

    result = classifier.test('test.txt')
    print 'P@1:', result.precision
    print 'R@1:', result.recall
    print 'Number of examples:', result.nexamples

This will print the same output to stdout as:

.. code:: shell

    ./fasttext test model.bin test.txt

In order to obtain the most likely label for a list of text, we can use
``classifer.predict`` method:

.. code:: python

    texts = ['example very long text 1', 'example very longtext 2']
    labels = classifier.predict(texts)
    print labels

    # Or with the probability
    labels = classifier.predict_proba(texts)
    print labels

We can specify ``k`` value to get the k-best labels from classifier:

.. code:: python

    labels = classifier.predict(texts, k=3)
    print labels

    # Or with the probability
    labels = classifier.predict_proba(texts, k=3)
    print labels

This interface is equivalent as ``fasttext(1)`` predict command. The
same model with the same input set will have the same prediction.

API documentation
-----------------

Skipgram model
~~~~~~~~~~~~~~

Train & load skipgram model

.. code:: python

    model = fasttext.skipgram(params)

List of available ``params`` and their default value:

::

    input_file     training file path (required)
    output         output file path (required)
    lr             learning rate [0.05]
    lr_update_rate change the rate of updates for the learning rate [100]
    dim            size of word vectors [100]
    ws             size of the context window [5]
    epoch          number of epochs [5]
    min_count      minimal number of word occurences [5]
    neg            number of negatives sampled [5]
    word_ngrams    max length of word ngram [1]
    loss           loss function {ns, hs, softmax} [ns]
    bucket         number of buckets [2000000]
    minn           min length of char ngram [3]
    maxn           max length of char ngram [6]
    thread         number of threads [12]
    t              sampling threshold [0.0001]
    silent         disable the log output from the C++ extension [1]
    encoding       specify input_file encoding [utf-8]

Example usage:

.. code:: python

    model = fasttext.skipgram('train.txt', 'model', lr=0.1, dim=300)

CBOW model
~~~~~~~~~~

Train & load CBOW model

.. code:: python

    model = fasttext.cbow(params)

List of available ``params`` and their default value:

::

    input_file     training file path (required)
    output         output file path (required)
    lr             learning rate [0.05]
    lr_update_rate change the rate of updates for the learning rate [100]
    dim            size of word vectors [100]
    ws             size of the context window [5]
    epoch          number of epochs [5]
    min_count      minimal number of word occurences [5]
    neg            number of negatives sampled [5]
    word_ngrams    max length of word ngram [1]
    loss           loss function {ns, hs, softmax} [ns]
    bucket         number of buckets [2000000]
    minn           min length of char ngram [3]
    maxn           max length of char ngram [6]
    thread         number of threads [12]
    t              sampling threshold [0.0001]
    silent         disable the log output from the C++ extension [1]
    encoding       specify input_file encoding [utf-8]

Example usage:

.. code:: python

    model = fasttext.cbow('train.txt', 'model', lr=0.1, dim=300)

Load pre-trained model
~~~~~~~~~~~~~~~~~~~~~~

File ``.bin`` that previously trained or generated by fastText can be
loaded using this function

.. code:: python

    model = fasttext.load_model('model.bin', encoding='utf-8')

Attributes and methods for the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skipgram and CBOW model have the following atributes & methods

.. code:: python

    model.model_name       # Model name
    model.words            # List of words in the dictionary
    model.dim              # Size of word vector
    model.ws               # Size of context window
    model.epoch            # Number of epochs
    model.min_count        # Minimal number of word occurences
    model.neg              # Number of negative sampled
    model.word_ngrams      # Max length of word ngram
    model.loss_name        # Loss function name
    model.bucket           # Number of buckets
    model.minn             # Min length of char ngram
    model.maxn             # Max length of char ngram
    model.lr_update_rate   # Rate of updates for the learning rate
    model.t                # Value of sampling threshold
    model.encoding         # Encoding of the model
    model[word]            # Get the vector of specified word

Supervised model
~~~~~~~~~~~~~~~~

Train & load the classifier

.. code:: python

    classifier = fasttext.supervised(params)

List of available ``params`` and their default value:

::

    input_file             training file path (required)
    output                 output file path (required)
    label_prefix           label prefix ['__label__']
    lr                     learning rate [0.1]
    lr_update_rate         change the rate of updates for the learning rate [100]
    dim                    size of word vectors [100]
    ws                     size of the context window [5]
    epoch                  number of epochs [5]
    min_count              minimal number of word occurences [1]
    neg                    number of negatives sampled [5]
    word_ngrams            max length of word ngram [1]
    loss                   loss function {ns, hs, softmax} [softmax]
    bucket                 number of buckets [0]
    minn                   min length of char ngram [0]
    maxn                   max length of char ngram [0]
    thread                 number of threads [12]
    t                      sampling threshold [0.0001]
    silent                 disable the log output from the C++ extension [1]
    encoding               specify input_file encoding [utf-8]
    pretrained_vectors	   pretrained word vectors (.vec file) for supervised learning []

Example usage:

.. code:: python

    classifier = fasttext.supervised('train.txt', 'model', label_prefix='__myprefix__',
                                     thread=4)

Load pre-trained classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~

File ``.bin`` that previously trained or generated by fastText can be
loaded using this function.

.. code:: shell

    ./fasttext supervised -input train.txt -output classifier -label 'some_prefix'

.. code:: python

    classifier = fasttext.load_model('classifier.bin', label_prefix='some_prefix')

Test classifier
~~~~~~~~~~~~~~~

This is equivalent as ``fasttext(1)`` test command. The test using the
same model and test set will produce the same value for the precision at
one and the number of examples.

.. code:: python

    result = classifier.test(params)

    # Properties
    result.precision # Precision at one
    result.recall    # Recall at one
    result.nexamples # Number of test examples

The param ``k`` is optional, and equal to ``1`` by default.

Predict the most-likely label of texts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This interface is equivalent as ``fasttext(1)`` predict command.

``texts`` is an array of string

.. code:: python

    labels = classifier.predict(texts, k)

    # Or with probability
    labels = classifier.predict_proba(texts, k)

The param ``k`` is optional, and equal to ``1`` by default.

Attributes and methods for the classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classifier have the following atributes & methods

.. code:: python

    classifier.labels                  # List of labels
    classifier.label_prefix            # Prefix of the label
    classifier.dim                     # Size of word vector
    classifier.ws                      # Size of context window
    classifier.epoch                   # Number of epochs
    classifier.min_count               # Minimal number of word occurences
    classifier.neg                     # Number of negative sampled
    classifier.word_ngrams             # Max length of word ngram
    classifier.loss_name               # Loss function name
    classifier.bucket                  # Number of buckets
    classifier.minn                    # Min length of char ngram
    classifier.maxn                    # Max length of char ngram
    classifier.lr_update_rate          # Rate of updates for the learning rate
    classifier.t                       # Value of sampling threshold
    classifier.encoding                # Encoding that used by classifier
    classifier.test(filename, k)       # Test the classifier
    classifier.predict(texts, k)       # Predict the most likely label
    classifier.predict_proba(texts, k) # Predict the most likely label include their probability

The param ``k`` for ``classifier.test``, ``classifier.predict`` and
``classifier.predict_proba`` is optional, and equal to ``1`` by default.

References
----------

Enriching Word Vectors with Subword Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[1] P. Bojanowski\*, E. Grave\*, A. Joulin, T. Mikolov, `*Enriching Word
Vectors with Subword
Information* <https://arxiv.org/pdf/1607.04606v1.pdf>`__

::

    @article{bojanowski2016enriching,
      title={Enriching Word Vectors with Subword Information},
      author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
      journal={arXiv preprint arXiv:1607.04606},
      year={2016}
    }

Bag of Tricks for Efficient Text Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[2] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, `*Bag of Tricks for
Efficient Text
Classification* <https://arxiv.org/pdf/1607.01759v2.pdf>`__

::

    @article{joulin2016bag,
      title={Bag of Tricks for Efficient Text Classification},
      author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
      journal={arXiv preprint arXiv:1607.01759},
      year={2016}
    }

(\* These authors contributed equally.)

Join the fastText community
---------------------------

-  Facebook page: https://www.facebook.com/groups/1174547215919768
-  Google group:
   https://groups.google.com/forum/#!forum/fasttext-library

.. |Build Status| image:: https://travis-ci.org/salestock/fastText.py.svg?branch=master
   :target: https://travis-ci.org/salestock/fastText.py
.. |PyPI version| image:: https://badge.fury.io/py/fasttext.svg
   :target: https://badge.fury.io/py/fasttext
