import fasttext

INPUT_TXT = '/path/to/file.txt'
OUTPUT_PATH_SKIPGRAM = '/tmp/skipgram'
OUTPUT_PATH_CBOW = '/tmp/cbow'

# Learn the word representation using skipgram model
skipgram = fasttext.skipgram(INPUT_TXT, OUTPUT_PATH, lr=0.02, dim=300, ws=5,
        epoch=1, min_count=5, neg=5, loss='ns', bucket=2000000, minn=3, maxn=6,
        thread=4, t=1e-4, lr_update_rate=100)

# Get the vector of some word
print skipgram['word']

# Learn the word representation using cbow model
cbow = fasttext.cbow(INPUT_TXT, OUTPUT_PATH, lr=0.02, dim=300, ws=5,
        epoch=1, min_count=5, neg=5, loss='ns', bucket=2000000, minn=3, maxn=6,
        thread=4, t=1e-4, lr_update_rate=100)

# Get the vector of some word
print cbow['word']

# Load pre-trained skipgram model
SKIPGRAM_BIN = OUTPUT_PATH_SKIPGRAM + '.bin'
skipgram = fasttext.load_model(SKIPGRAM_BIN)
print skipgram['word']

# Load pre-trained cbow model
CBOW_BIN = OUTPUT_PATH_CBOW + '.bin'
cbow = fasttext.load_model(CBOW_BIN)
print cbow['word']

