
all: install test

test: test-skipgram test-cbow test-classifier

buildext:
	python setup.py build_ext --inplace
.PHONY: buildext

install:
	pip install -r requirements.txt
	python setup.py install
.PHONY: install

# Install the pandoc(1) first to run this command
# sudo apt-get install pandoc
README.rst: README.md
	pandoc --from=markdown --to=rst --output=README.rst README.md

upload: README.rst
	python setup.py sdist upload

upload-to-pypitest: README.rst
	python setup.py sdist upload -r pypitest
.PHONY: upload-to-pypitest

install-from-pypitest::
	pip install -U --no-cache-dir -i https://testpypi.python.org/pypi fasttext
.PHONY: install-from-pypitest

install-dev: README.rst
	python setup.py develop
.PHONY: install-dev

pre-test:
	# Remove generated file from test
	rm -f test/*.vec test/*.bin test/*_result.txt
.PHONY: pre-test

fasttext/cpp/fasttext:
	rm -f fasttext/cpp/fasttext
	make --directory fasttext/cpp/

# Test for skipgram model
# Redirect stdout to /dev/null to prevent exceed the log limit size from
# Travis CI
test/skipgram_params_test.bin:
	./fasttext/cpp/fasttext skipgram -input test/params_test.txt -output \
		test/skipgram_params_test -lr 0.025 -dim 100 -ws 5 -epoch 1 \
		-minCount 1 -neg 5 -loss ns -bucket 2000000 -minn 3 -maxn 6 \
		-thread 4 -lrUpdateRate 100 -t 1e-4 >> /dev/null

# Generate default value of skipgram command from fasttext(1)
test/skipgram_default_params_result.txt:
	$(MAKE) skipgram_default_params_result.txt --directory test/

test-skipgram: pre-test fasttext/cpp/fasttext test/skipgram_params_test.bin \
			   test/skipgram_default_params_result.txt
	python test/skipgram_test.py --verbose

# Test for cbow model
# Redirect stdout to /dev/null to prevent exceed the log limit size from
# Travis CI
test/cbow_params_test.bin:
	./fasttext/cpp/fasttext cbow -input test/params_test.txt -output \
		test/cbow_params_test -lr 0.005 -dim 50 -ws 5 -epoch 1 \
		-minCount 1 -neg 5 -loss ns -bucket 2000000 -minn 3 -maxn 6 \
		-thread 4 -lrUpdateRate 100 -t 1e-4 >> /dev/null

# Generate default value of cbow command from fasttext(1)
test/cbow_default_params_result.txt:
	$(MAKE) cbow_default_params_result.txt --directory test/

test-cbow: pre-test fasttext/cpp/fasttext test/cbow_params_test.bin \
		   test/cbow_default_params_result.txt
	python test/cbow_test.py --verbose

# Test for classifier
test/dbpedia.train: test/download_dbpedia.sh
	sh test/download_dbpedia.sh # Download & normalize training file

# Redirect stdout to /dev/null to prevent exceed the log limit size from
# Travis CI
test/classifier.bin: test/dbpedia.train
	./fasttext/cpp/fasttext supervised -input test/dbpedia.train \
		-output test/classifier -dim 100 -lr 0.1 -wordNgrams 2 \
		-minCount 1 -bucket 2000000 -epoch 5 -thread 4 >> /dev/null

test/classifier_test_result.txt: test/classifier.bin
	./fasttext/cpp/fasttext test test/classifier.bin \
		test/classifier_test.txt > test/classifier_test_result.txt

test/classifier_pred_result.txt: test/classifier.bin
	./fasttext/cpp/fasttext predict test/classifier.bin \
		test/classifier_pred_test.txt > \
		test/classifier_pred_result.txt

test/classifier_pred_k_result.txt: test/classifier.bin
	./fasttext/cpp/fasttext predict test/classifier.bin \
		test/classifier_pred_test.txt 5 > \
		test/classifier_pred_k_result.txt

test/classifier_pred_prob_result.txt: test/classifier.bin
	./fasttext/cpp/fasttext predict-prob test/classifier.bin \
		test/classifier_pred_test.txt > \
		test/classifier_pred_prob_result.txt

test/classifier_pred_prob_k_result.txt: test/classifier.bin
	./fasttext/cpp/fasttext predict-prob test/classifier.bin \
		test/classifier_pred_test.txt 5 > \
		test/classifier_pred_prob_k_result.txt

# Generate default value of classifier command from fasttext(1)
test/classifier_default_params_result.txt:
	$(MAKE) classifier_default_params_result.txt --directory test/

test-classifier: pre-test fasttext/cpp/fasttext test/classifier.bin \
				 test/classifier_test_result.txt \
				 test/classifier_pred_result.txt \
				 test/classifier_pred_k_result.txt \
				 test/classifier_pred_prob_result.txt \
				 test/classifier_pred_prob_k_result.txt \
				 test/classifier_default_params_result.txt
	python test/classifier_test.py --verbose

