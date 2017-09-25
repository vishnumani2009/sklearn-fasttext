import fasttext as ft

# First download the dbpedia.train using https://github.com/facebookresearch/fastText/blob/master/classification-example.sh
# on test/ and move to the example directory
current_dir = path.dirname(__file__)
input_file = path.join(current_dir, 'dbpedia.train')
output = '/tmp/classifier'
test_file = '../test/classifier_test.txt'

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
classifier = ft.supervised(input_file, output, dim=dim, lr=lr, epoch=epoch,
    min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,
    thread=thread, silent=silent, label_prefix=label_prefix)

# Test the classifier
result = classifier.test(test_file)
print 'P@1:', result.precision
print 'R@1:', result.recall
print 'Number of examples:', result.nexamples

# Predict some text
# (Example text is from dbpedia.train)
texts = ['birchas chaim , yeshiva birchas chaim is a orthodox jewish mesivta \
        high school in lakewood township new jersey . it was founded by rabbi \
        shmuel zalmen stein in 2001 after his father rabbi chaim stein asked \
        him to open a branch of telshe yeshiva in lakewood . as of the 2009-10 \
        school year the school had an enrollment of 76 students and 6 . 6 \
        classroom teachers ( on a fte basis ) for a student–teacher ratio of \
        11 . 5 1 .']
labels = classifier.predict(texts)
print labels
