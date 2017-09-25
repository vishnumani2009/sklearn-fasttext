/* To print the value of default params from fasttext(1) */
#include <iostream>

#include "../fasttext/cpp/src/args.h"

using namespace fasttext;

int main(int argc, char **argv)
{
    Args args;
    args.parseArgs(argc, argv);
    std::cout << "lr " << args.lr << std::endl;
    std::cout << "dim " << args.dim << std::endl;
    std::cout << "ws " << args.ws << std::endl;
    std::cout << "epoch " << args.epoch << std::endl;
    std::cout << "minCount " << args.minCount << std::endl;
    std::cout << "neg " << args.neg << std::endl;
    std::cout << "wordNgrams " << args.wordNgrams << std::endl;
    std::string lossName;
    if(args.loss == loss_name::ns) {
        lossName = "ns";
    }
    if(args.loss == loss_name::hs) {
        lossName = "hs";
    }
    if(args.loss == loss_name::softmax) {
        lossName = "softmax";
    }
    std::cout << "loss " << lossName << std::endl;
    std::cout << "bucket " << args.bucket << std::endl;
    std::cout << "minn " << args.minn << std::endl;
    std::cout << "maxn " << args.maxn << std::endl;
    std::cout << "thread " << args.thread << std::endl;
    std::cout << "lrUpdateRate " << args.lrUpdateRate << std::endl;
    std::cout << "t " << args.t << std::endl;
    std::cout << "label " << args.label << std::endl;
}
