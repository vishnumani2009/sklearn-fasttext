/* An interface for fastText */
#include <streambuf>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "interface.h"
#include "cpp/src/real.h"
#include "cpp/src/args.h"
#include "cpp/src/dictionary.h"
#include "cpp/src/matrix.h"
#include "cpp/src/vector.h"
#include "cpp/src/model.h"
#include "cpp/src/fasttext.h"

FastTextModel::FastTextModel(){}

std::vector<std::string> FastTextModel::getWords()
{
    return _words;
}

void FastTextModel::addWord(std::string word)
{
    _words.push_back(word);
}

void FastTextModel::setArgs(std::shared_ptr<Args> args)
{
    dim = args->dim;
    ws = args->ws;
    epoch = args->epoch;
    minCount = args->minCount;
    neg = args->neg;
    wordNgrams = args->wordNgrams;
    if(args->loss == loss_name::ns) {
        lossName = "ns";
    }
    if(args->loss == loss_name::hs) {
        lossName = "hs";
    }
    if(args->loss == loss_name::softmax) {
        lossName = "softmax";
    }
    if(args->model == model_name::cbow) {
        modelName = "cbow";
    }
    if(args->model == model_name::sg) {
        modelName = "skipgram";
    }
    if(args->model == model_name::sup) {
        modelName = "supervised";
    }
    bucket = args->bucket;
    minn = args->minn;
    maxn = args->maxn;
    lrUpdateRate = args->lrUpdateRate;
    t = args->t;
    lr = args->lr;
}

void FastTextModel::setDictionary(std::shared_ptr<Dictionary> dict)
{
    _dict = dict;
}

void FastTextModel::setMatrix(std::shared_ptr<Matrix> input,
        std::shared_ptr<Matrix> output)
{
    _input_matrix = input;
    _output_matrix = output;
}

void FastTextModel::setModel(std::shared_ptr<Model> model)
{
    _model = model;
}

/* Methods to wrap the Dictionary methods; since we can't access
 * dicrectly Dictionary in python because Dictionary doesn't have
 * nullary constructor */
int32_t FastTextModel::dictGetNWords()
{
    return _dict->nwords();
}

std::string FastTextModel::dictGetWord(int32_t i)
{
    return _dict->getWord(i);
}

int32_t FastTextModel::dictGetNLabels()
{
    return _dict->nlabels();
}

std::string FastTextModel::dictGetLabel(int32_t i)
{
    return _dict->getLabel(i);
}

/* We use the same logic as FastText::getVector here; Because
 * we need to access our own dictionary and input matrix */
std::vector<real> FastTextModel::getVectorWrapper(std::string word)
{
    Vector vec(dim);
    const std::vector<int32_t>& ngrams = _dict->getNgrams(word);
    vec.zero();
    for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
        vec.addRow(*_input_matrix, *it);
    }
    if (ngrams.size() > 0) {
        vec.mul(1.0 / ngrams.size());
    }
    std::vector<real> vector(vec.data_, vec.data_ + vec.m_);
    return vector;
}

std::vector<double> FastTextModel::classifierTest(std::string filename,
        int32_t k)
{
    int32_t nexamples = 0;
    int32_t nlabels = 0;
    double precision = 0.0;
    std::vector<int32_t> line;
    std::vector<int32_t> labels;
    std::ifstream ifs(filename);
    if(!ifs.is_open()) {
        std::cerr << "interface.cc: Test file cannot be opened!" << std::endl;
        exit(EXIT_FAILURE);
    }

    while (ifs.peek() != EOF) {
        _dict->getLine(ifs, line, labels, _model->rng);
        _dict->addNgrams(line, wordNgrams);
        if(labels.size() > 0 && line.size() > 0) {
            std::vector<std::pair<real, int32_t>> predictions;
            _model->predict(line, k, predictions);
            for(auto it = predictions.cbegin(); it != predictions.cend();
                    it++) {
                int32_t i = it->second;
                if(std::find(labels.begin(), labels.end(), i)
                        != labels.end()) {
                    precision += 1.0;
                }
            }
            nexamples++;
            nlabels += labels.size();
        }
    }

    ifs.close();
    std::setprecision(3);
    std::vector<double> result;
    result.push_back(precision/(k * nexamples));
    result.push_back(precision/nlabels);
    result.push_back((double)nexamples);
    return result;
}

std::vector<std::string> FastTextModel::classifierPredict(std::string text,
        int32_t k)
{
    /* Hardcoded here; since we need this variable but the variable
     * is private in dictionary.h */
    const int32_t max_line_size = 1024;

    /* List of word ids */
    std::vector<int32_t> text_word_ids;
    std::istringstream iss(text);
    std::string token;

    /* We implement the same logic as Dictionary::getLine */
    std::uniform_real_distribution<> uniform(0, 1);
    while(_dict->readWord(iss, token)) {
        int32_t word_id = _dict->getId(token);
        if(word_id < 0) continue;
        entry_type type = _dict->getType(word_id);
        if (type == entry_type::word &&
                !_dict->discard(word_id, uniform(_model->rng))) {
            text_word_ids.push_back(word_id);
        }
        if(text_word_ids.size() > max_line_size) break;
    }
    _dict->addNgrams(text_word_ids, wordNgrams);

    std::vector<std::string> labels;
    if(text_word_ids.size() > 0) {
        std::vector<std::pair<real, int32_t>> predictions;

        _model->predict(text_word_ids, k, predictions);
        for(auto it = predictions.cbegin(); it != predictions.cend(); it++) {
            labels.push_back(_dict->getLabel(it->second));
        }

        return labels;
    } else {
        return labels;
    }
}

std::vector<std::vector<std::string>>
    FastTextModel::classifierPredictProb(std::string text, int32_t k)
{
    /* Hardcoded here; since we need this variable but the variable
     * is private in dictionary.h */
    const int32_t max_line_size = 1024;

    /* List of word ids */
    std::vector<int32_t> text_word_ids;
    std::istringstream iss(text);
    std::string token;

    /* We implement the same logic as Dictionary::getLine */
    std::uniform_real_distribution<> uniform(0, 1);
    while(_dict->readWord(iss, token)) {
        int32_t word_id = _dict->getId(token);
        if(word_id < 0) continue;
        entry_type type = _dict->getType(word_id);
        if (type == entry_type::word &&
                !_dict->discard(word_id, uniform(_model->rng))) {
            text_word_ids.push_back(word_id);
        }
        if(text_word_ids.size() > max_line_size) break;
    }
    _dict->addNgrams(text_word_ids, wordNgrams);

    std::vector<std::vector<std::string>> results;
    if(text_word_ids.size() > 0) {
        std::vector<std::pair<real, int32_t>> predictions;

        _model->predict(text_word_ids, k, predictions);
        for(auto it = predictions.cbegin(); it != predictions.cend(); it++) {
            std::vector<std::string> result;
            result.push_back(_dict->getLabel(it->second));

            /* We use string stream here instead of to_string, to make sure
             * that the string is consistent with std::cout from fasttext(1) */
            std::ostringstream probability_stream;
            probability_stream << exp(it->first);
            result.push_back(probability_stream.str());

            results.push_back(result);
        }
    }
    return results;
}

template <class cT, class traits = std::char_traits<cT> >
class basic_nullbuf: public std::basic_streambuf<cT, traits> {
    typename traits::int_type overflow(typename traits::int_type c)
    {
        return traits::not_eof(c); // indicate success
    }
};

void trainWrapper(int argc, char **argv, int silent)
{
    /* if silent > 0, the log from train() function will be supressed */
    if(silent > 0) {
        /* output file stream to redirect output from fastText library */
        std::streambuf* old_ofs = std::cout.rdbuf();
        std::streambuf* null_ofs = new basic_nullbuf<char>();
        std::cout.rdbuf(null_ofs);
        std::shared_ptr<Args> a = std::make_shared<Args>();
        a->parseArgs(argc, argv);
        FastText fasttext;
        fasttext.train(a);
        std::cout.rdbuf(old_ofs);
        delete null_ofs;
    } else {
        std::shared_ptr<Args> a = std::make_shared<Args>();
        a->parseArgs(argc, argv);
        FastText fasttext;
        fasttext.train(a);
    }
}

/* The logic is the same as FastText::loadModel, we roll our own
 * to be able to access data from args, dictionary etc since this
 * data is private in FastText class */
void loadModelWrapper(std::string filename, FastTextModel& model)
{
    std::ifstream ifs(filename, std::ios_base::in | std::ios_base::binary);
    if (!ifs.is_open()) {
        std::cerr << "interface.cc: cannot load model file ";
        std::cerr << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::shared_ptr<Args> args = std::make_shared<Args>();
    std::shared_ptr<Dictionary> dict = std::make_shared<Dictionary>(args);
    std::shared_ptr<Matrix> input_matrix = std::make_shared<Matrix>();
    std::shared_ptr<Matrix> output_matrix = std::make_shared<Matrix>();
    args->load(ifs);
    dict->load(ifs);
    input_matrix->load(ifs);
    output_matrix->load(ifs);
    std::shared_ptr<Model> model_p = std::make_shared<Model>(input_matrix,
            output_matrix, args, 0);
    if (args->model == model_name::sup) {
        model_p->setTargetCounts(dict->getCounts(entry_type::label));
    } else {
        model_p->setTargetCounts(dict->getCounts(entry_type::word));
    }
    ifs.close();

    /* save all data to FastTextModel */
    model.setArgs(args);
    model.setDictionary(dict);
    model.setMatrix(input_matrix, output_matrix);
    model.setModel(model_p);
}
