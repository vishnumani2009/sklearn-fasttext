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
skipgram_file = path.join(test_dir, 'skipgram_params_test.bin')
input_file = path.join(test_dir, 'params_test.txt')
output = path.join(test_dir, 'generated_skipgram')
params_txt = path.join(test_dir, 'skipgram_default_params_result.txt')

# Test to make sure that skipgram interface run correctly
class TestSkipgramModel(unittest.TestCase):
    def test_load_skipgram_model(self):
        model = ft.load_model(skipgram_file, encoding='utf-8')

        # Make sure the model is returned correctly
        self.assertEqual(model.model_name, 'skipgram')

        # Make sure all params loaded correctly
        # see Makefile on target test-skipgram for the params
        self.assertEqual(model.dim, 100)
        self.assertEqual(model.ws, 5)
        self.assertEqual(model.epoch, 1)
        self.assertEqual(model.min_count, 1)
        self.assertEqual(model.neg, 5)
        self.assertEqual(model.loss_name, 'ns')
        self.assertEqual(model.bucket, 2000000)
        self.assertEqual(model.minn, 3)
        self.assertEqual(model.maxn, 6)
        self.assertEqual(model.lr_update_rate, 100)
        self.assertEqual(model.t, 1e-4)

        # Make sure the vector have the right dimension
        self.assertEqual(len(model['the']), model.dim)

        # Make sure we support unicode character
        unicode_str = 'Καλημέρα'
        self.assertTrue(unicode_str in model.words)
        self.assertEqual(len(model[unicode_str]), model.dim)

    def test_load_invalid_skipgram_model(self):
        # Make sure we are throwing an exception
        self.assertRaises(ValueError, ft.load_model, '/path/to/invalid')

    def test_train_skipgram_model_default(self):
        default_args = default_params.read_file(params_txt)
        model = ft.skipgram(input_file, output)

        # Make sure the default params of skipgram is equal
        # to fasttext(1) default params
        self.assertEqual(model.model_name, 'skipgram')
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

    def test_train_skipgram_model(self):
        # set params
        lr=0.005
        dim=10
        ws=5
        epoch=5
        min_count=1
        neg=5
        word_ngrams=1
        loss='ns'
        bucket=2000000
        minn=3
        maxn=6
        thread=4
        lr_update_rate=10000
        t=1e-4
        silent=1

        # train skipgram model
        model = ft.skipgram(input_file, output, lr, dim, ws, epoch, min_count,
                neg, word_ngrams, loss, bucket, minn, maxn, thread, lr_update_rate,
                t, silent)

        # Make sure the model is generated correctly
        self.assertEqual(model.dim, dim)
        self.assertEqual(model.ws, ws)
        self.assertEqual(model.epoch, epoch)
        self.assertEqual(model.min_count, min_count)
        self.assertEqual(model.neg, neg)
        self.assertEqual(model.loss_name, loss)
        self.assertEqual(model.bucket, bucket)
        self.assertEqual(model.minn, minn)
        self.assertEqual(model.maxn, maxn)
        self.assertEqual(model.lr_update_rate, lr_update_rate)
        self.assertEqual(model.t, t)

        # Make sure .bin and .vec are generated
        self.assertTrue(path.isfile(output + '.bin'))
        self.assertTrue(path.isfile(output + '.vec'))

        # Make sure the vector have the right dimension
        self.assertEqual(len(model['the']), dim)

        # Make sure we support unicode character
        unicode_str = 'Καλημέρα'
        self.assertTrue(unicode_str in model.words)
        self.assertTrue(unicode_str in model)
        self.assertEqual(len(model[unicode_str]), model.dim)

if __name__ == '__main__':
    unittest.main()
