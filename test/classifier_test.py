# Set encoding to support Python 2
# -*- coding: utf-8 -*-

# We use unicode_literals to generalize unicode syntax in plain string ''
# instead of u''. (to support python 3.2)
from __future__ import unicode_literals
import unittest
from os import path

import fasttext as ft

import default_params

test_dir = path.dirname(__file__)
classifier_bin = path.join(test_dir, 'classifier.bin')
input_file = path.join(test_dir, 'dbpedia.train')
pred_file  = path.join(test_dir, 'classifier_pred_test.txt')
output = path.join(test_dir, 'generated_classifier')
test_result = path.join(test_dir, 'classifier_test_result.txt')
pred_result = path.join(test_dir, 'classifier_pred_result.txt')
pred_k_result = path.join(test_dir, 'classifier_pred_k_result.txt')
pred_prob_result = path.join(test_dir, 'classifier_pred_prob_result.txt')
pred_prob_k_result = path.join(test_dir, 'classifier_pred_prob_k_result.txt')
test_file = path.join(test_dir, 'classifier_test.txt')
params_txt = path.join(test_dir, 'classifier_default_params_result.txt')
pretrained_vectors_path = path.join(test_dir, 'generated_skipgram.vec')

# To validate model are loaded correctly
def read_labels_from_input(filename, label_prefix):
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            # Python 2 read file in ASCII encoding by default
            # so we need to decode the str to UTF-8 first.
            # But, in Python 3, str doesn't have decode method
            # so this decoding step make the test fails.
            # Python 3 read file in UTF-8 encoding by default so
            # we wrap this in the try-except to support both Python 2
            # and Python 3
            try:
                line = line.decode('utf-8')
            except:
                line = line

            label = line.split(',', 1)[0].strip()
            label = label.replace(label_prefix, '')
            if label in labels:
                continue
            else:
                labels.append(label)
    return labels

# To validate model have the same prediction as fasttext(1)
def read_labels_from_result(filename, label_prefix):
    all_labels = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                line = line.decode('utf-8')
            except:
                line = line

            labels = []
            raw_labels = line.split(' ')
            for raw_label in raw_labels:
                label = raw_label.replace(label_prefix, '')
                labels.append(label.strip())
            all_labels.append(labels)
    return all_labels

def read_labels_from_result_prob(filename, label_prefix):
    all_labels = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                line = line.decode('utf-8')
            except:
                line = line

            labels = []
            probabilities = []
            raw = line.split(' ')
            prefix_len = len(label_prefix)
            for w in raw:
                w = w.strip()
                if len(w) < prefix_len:
                    probabilities.append(float(w))
                elif w[:prefix_len] == label_prefix:
                    label = w.replace(label_prefix, '')
                    labels.append(label)
                else:
                    probabilities.append(float(w))
            all_labels.append(list(zip(labels, probabilities)))
    return all_labels

# To read text data to predict
def read_texts(pred_file):
    texts = []
    with open(pred_file, 'r') as f:
        for line in f:
            try:
                line = line.decode('utf-8')
            except:
                line = line
            texts.append(line)
    return texts

# Test to make sure that classifier interface run correctly
class TestClassifierModel(unittest.TestCase):
    def test_load_classifier_model(self):
        label_prefix='__label__'
        model = ft.load_model(classifier_bin, label_prefix=label_prefix,
                encoding='utf-8')

        # Make sure the model is returned correctly
        self.assertEqual(model.model_name, 'supervised')

        # Make sure all params loaded correctly
        # see Makefile on target test-supervised for the params
        self.assertEqual(model.dim, 100)
        self.assertEqual(model.word_ngrams, 2)
        self.assertEqual(model.min_count, 1)
        self.assertEqual(model.epoch, 5)
        self.assertEqual(model.bucket, 2000000)

        # Read labels from the the input_file
        labels = read_labels_from_input(input_file, label_prefix)

        # Make sure labels are loaded correctly
        self.assertTrue(sorted(model.labels) == sorted(labels))

    def test_load_invalid_classifier_model(self):
        # Make sure we are throwing an exception
        self.assertRaises(ValueError, ft.load_model, '/path/to/invalid',
                label_prefix='__label__')

    def test_train_classifier_model_default(self):
        default_args = default_params.read_file(params_txt)
        model = ft.supervised(input_file, output)

        # Make sure the default params of supervised is equal
        # to fasttext(1) default params
        self.assertEqual(model.model_name, 'supervised')
        self.assertEqual(model.dim, int(default_args['dim']))
        self.assertEqual(model.ws, int(default_args['ws']))
        self.assertEqual(model.epoch, int(default_args['epoch']))
        self.assertEqual(model.min_count, int(default_args['minCount']))
        self.assertEqual(model.neg, int(default_args['neg']))
        self.assertEqual(model.word_ngrams, int(default_args['wordNgrams']))
        self.assertEqual(model.loss_name, default_args['loss'])
        self.assertEqual(model.bucket, int(default_args['bucket']))
        self.assertEqual(model.minn, int(default_args['minn']))
        self.assertEqual(model.maxn, int(default_args['maxn']))
        self.assertEqual(model.lr_update_rate,
                float(default_args['lrUpdateRate']))
        self.assertEqual(model.t, float(default_args['t']))
        self.assertEqual(model.label_prefix, default_args['label'])

    def test_train_classifier(self):
        # set params
        dim=10
        lr=0.005
        epoch=1
        min_count=1
        word_ngrams=3
        bucket=2000000
        thread=4
        silent=1
        label_prefix='__label__'

        # Train the classifier
        model = ft.supervised(input_file, output, dim=dim, lr=lr, epoch=epoch,
                min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,
                thread=thread, silent=silent, label_prefix=label_prefix)

        # Make sure the model is generated correctly
        self.assertEqual(model.dim, dim)
        self.assertEqual(model.epoch, epoch)
        self.assertEqual(model.min_count, min_count)
        self.assertEqual(model.word_ngrams, word_ngrams)
        self.assertEqual(model.bucket, bucket)

        # Read labels from the the input_file
        labels = read_labels_from_input(input_file, label_prefix)

        # Make sure labels are loaded correctly
        self.assertTrue(sorted(model.labels) == sorted(labels))

        # Make sure .bin and .vec are generated
        self.assertTrue(path.isfile(output + '.bin'))

        # Test some methods, make sure it works
        labels = model.predict(['some long long texts'])
        self.assertTrue(type(labels) == type([]))
        labels = model.predict_proba(['some long long texts'])
        self.assertTrue(type(labels) == type([]))

    def test_train_classifier_pretrained_vectors(self):
        # set params
        dim=100
        lr=0.005
        epoch=1
        min_count=1
        word_ngrams=3
        bucket=2000000
        thread=4
        silent=1
        label_prefix='__label__'

        # Make sure the pretrained vectors exists
        self.assertTrue(path.isfile(pretrained_vectors_path), "The model used as pretrained vectors does not exist."
                                                              "Please ensure that skipgram tests ran before this one.")

        # Train the classifier
        model = ft.supervised(input_file, output, dim=dim, lr=lr, epoch=epoch,
                min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,
                thread=thread, silent=silent, label_prefix=label_prefix, pretrained_vectors=pretrained_vectors_path)

        # Make sure the model is generated correctly
        self.assertEqual(model.dim, dim)
        self.assertEqual(model.epoch, epoch)
        self.assertEqual(model.min_count, min_count)
        self.assertEqual(model.word_ngrams, word_ngrams)
        self.assertEqual(model.bucket, bucket)

        # Read labels from the the input_file
        labels = read_labels_from_input(input_file, label_prefix)

        # Make sure labels are loaded correctly
        self.assertTrue(sorted(model.labels) == sorted(labels))

        # Make sure .bin and .vec are generated
        self.assertTrue(path.isfile(output + '.bin'))

        # Test some methods, make sure it works
        labels = model.predict(['some long long texts'])
        self.assertTrue(type(labels) == type([]))
        labels = model.predict_proba(['some long long texts'])
        self.assertTrue(type(labels) == type([]))

    def test_classifier_test(self):
        # Read the test result from fasttext(1) using the same classifier model
        precision_at_one = 0.0
        nexamples = 0
        with open(test_result) as f:
            lines = f.readlines()
            precision_at_one = float(lines[0][5:].strip())
            recall_at_one = float(lines[1][5:].strip())
            nexamples = int(lines[2][20:].strip())

        # Load and test using the same model and test set
        classifier = ft.load_model(classifier_bin, label_prefix='__label__')
        result = classifier.test(test_file, k=1)

        # Make sure that the test result is the same as the result generated
        # by fasttext(1)
        p_at_1 = float("{0:.2f}".format(result.precision))
        r_at_1 = float("{0:.2f}".format(result.recall))
        self.assertEqual(p_at_1, precision_at_one)
        self.assertEqual(r_at_1, recall_at_one)
        self.assertEqual(result.nexamples, nexamples)

    def test_classifier_predict(self):
        # Load the pre-trained classifier
        label_prefix = '__label__'
        classifier = ft.load_model(classifier_bin, label_prefix=label_prefix)

        # Read prediction result from fasttext(1)
        fasttext_labels = read_labels_from_result(pred_result,
                label_prefix=label_prefix)

        # Read texts from the pred_file
        texts = read_texts(pred_file)

        # Predict the labels
        labels = classifier.predict(texts)

        # Make sure the returned labels are the same as predicted by
        # fasttext(1)
        self.assertTrue(labels == fasttext_labels)

    def test_classifier_predict_k_best(self):
        label_prefix = '__label__'
        # Load the pre-trained classifier
        classifier = ft.load_model(classifier_bin, label_prefix=label_prefix)

        # Read prediction result from fasttext(1)
        fasttext_labels = read_labels_from_result(pred_k_result,
                label_prefix=label_prefix)

        # Read texts from the pred_file
        texts = read_texts(pred_file)

        # Predict the k-best labels
        labels = classifier.predict(texts, k=5)

        # Make sure the returned labels are the same as predicted by
        # fasttext(1)
        self.assertTrue(labels == fasttext_labels)

    def test_classifier_predict_prob(self):
        # Load the pre-trained classifier
        label_prefix = '__label__'
        classifier = ft.load_model(classifier_bin, label_prefix=label_prefix)

        # Read prediction result from fasttext(1)
        fasttext_labels = read_labels_from_result_prob(pred_prob_result,
                label_prefix=label_prefix)

        # Read texts from the pred_file
        texts = read_texts(pred_file)

        # Predict the labels
        labels = classifier.predict_proba(texts)

        # Make sure the returned labels are the same as predicted by
        # fasttext(1)
        self.assertTrue(labels == fasttext_labels)

    def test_classifier_predict_prob_k_best(self):
        label_prefix = '__label__'
        # Load the pre-trained classifier
        classifier = ft.load_model(classifier_bin, label_prefix=label_prefix)

        # Read prediction result from fasttext(1)
        fasttext_labels = read_labels_from_result_prob(pred_prob_k_result,
                label_prefix=label_prefix)

        # Read texts from the pred_file
        texts = read_texts(pred_file)

        # Predict the k-best labels
        labels = classifier.predict_proba(texts, k=5)

        # Make sure the returned labels are the same as predicted by
        # fasttext(1)
        self.assertTrue(labels == fasttext_labels)

if __name__ == '__main__':
    unittest.main()

