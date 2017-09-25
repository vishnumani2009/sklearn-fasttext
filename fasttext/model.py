# fastText Model representation in Python
import numpy as np
from numpy.linalg import norm

# Class for Skipgram and CBOW model
class WordVectorModel(object):
    def __init__(self, model, words, encoding='utf-8'):
        self._model = model
        self.words = set(words)
        self.dim = model.dim
        self.ws = model.ws
        self.epoch = model.epoch
        self.min_count = model.minCount
        self.neg = model.neg
        self.word_ngrams = model.wordNgrams
        self.loss_name = model.lossName.decode(encoding)
        self.model_name = model.modelName.decode(encoding)
        self.bucket = model.bucket
        self.minn = model.minn
        self.maxn = model.maxn
        self.lr_update_rate = model.lrUpdateRate
        self.t = model.t
        self.encoding = encoding

    def __getitem__(self, word):
        return self._model.get_vector(word, self.encoding)

    def __contains__(self, word):
        return word in self.words

    def cosine_similarity(self, first_word, second_word):
        v1 = self.__getitem__(first_word)
        v2 = self.__getitem__(second_word)
        dot_product = np.dot(v1, v2)
        cosine_sim = dot_product / (norm(v1) * norm(v2))
        return cosine_sim

# Class for classifier model
class SupervisedModel(object):
    def __init__(self, model, labels, label_prefix, encoding='utf-8'):
        self._model = model
        self.labels = labels
        self.dim = model.dim
        self.ws = model.ws
        self.epoch = model.epoch
        self.min_count = model.minCount
        self.neg = model.neg
        self.word_ngrams = model.wordNgrams
        self.loss_name = model.lossName.decode(encoding)
        self.model_name = model.modelName.decode(encoding)
        self.bucket = model.bucket
        self.minn = model.minn
        self.maxn = model.maxn
        self.lr_update_rate = model.lrUpdateRate
        self.t = model.t
        self.label_prefix = label_prefix
        self.encoding = encoding

    def test(self, test_file, k=1):
        return self._model.classifier_test(test_file, k, self.encoding)

    def predict(self, texts, k=1):
        all_labels = []
        for text in texts:
            if text[-1] != '\n':
                text += '\n'
            labels = self._model.classifier_predict(text, k,
                    self.label_prefix, self.encoding)
            all_labels.append(labels)
        return all_labels

    def predict_proba(self, texts, k=1):
        results = []
        for text in texts:
            if text[-1] != '\n':
                text += '\n'
            result = self._model.classifier_predict_prob(text, k,
                    self.label_prefix, self.encoding)
            results.append(result)
        return results

# Class for test result
class ClassifierTestResult(object):
    def __init__(self, precision, recall, nexamples):
        self.precision = precision
        self.recall = recall
        self.nexamples = nexamples

