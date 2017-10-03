from sklearn.base import BaseEstimator, ClassifierMixin
import fasttext

class FastTextClassifier(BaseEstimator,ClassifierMixin):
"""Base classiifer of Fasttext estimator"""

	def __init__(self,lpr='__label__',lr=0.1,lru=100,dim=100,ws=5,epoch=5,minc=1,neg=5,ngram=1,loss='softmax',nbucket=0,minn=0,maxn=0):
		"""
		label_prefix   			label prefix ['__label__']
		lr             			learning rate [0.1]
		lr_update_rate 			change the rate of updates for the learning rate [100]
		dim            			size of word vectors [100]
		ws             			size of the context window [5]
		epoch          			number of epochs [5]
		min_count      			minimal number of word occurences [1]
		neg            			number of negatives sampled [5]
		word_ngrams    			max length of word ngram [1]
		loss           			loss function {ns, hs, softmax} [softmax]
		bucket         			number of buckets [0]
		minn           			min length of char ngram [0]
		maxn           			min length of char ngram [0]
		"""
		self.label_prefix=lpr
		self.lr=lr
		self.lr_update_rate=lru
		self.dim=dim
		self.ws=ws
		self.epoch=epoch
		self.min_count=minc
		self.neg=neg
		self.word_ngrams=ngram
		self.loss=loss
		self.bucket=bucket
		self.minn=minn
		self.maxn=maxn

	def fit(self,X,y):
		'train a classifier and return the model'
		pass
	def predict(self,X):
		'get class labels for the classifier'
		pass
	def report(self,model=None):
		'prints classification report'
		pass
	def predict_proba(self,X,y):
		'return predicted probabilities'
		pass
	def getlabels(self):
		'retuns classlabels'
		pass
	def getproperties(self):
                '''
                Input: Nothing, other than object self pointer
                Return: None , prints the descriptions of the model hyperparameters
                '''
                print("The model has following hyperparameters as part of its specification")
                print("Label prefix used : "+str(self.label_prefix)
                print("Learning rate :"+ str(lr))
                print("Learning rate update after "+str(self.lr_update_rate)+"iterations")
                print("Embedding size: "+str(self.dim))
                print("Epochs :"+ str(self.epochs)
                print("minimal number of word occurences: "+self.min_count)
                print("number of negatives sampled :"+str(self.neg))
                print("max length of word ngram "+str(self.word_ngrams))
                print("loss function: "+str(self.loss))
                print("number of buckets "+str(self.bucket))
                print("min length of char ngram:"+str(self.minn))
                print("min length of char ngram"+ str(self.maxn))
                return(None)
                
		
	def loadpretrained(self,X):
		'returns the model with pretrained weights'
		pass
	
class SkipgramFastText(BaseEstimator,ClassifierMixin):

	def __init__(self,lpr='__label__',lr=0.1,lru=100,dim=100,ws=5,epoch=5,minc=1,neg=5,ngram=1,\
loss='softmax',nbucket=0,minn=0,maxn=0,th=12,t=0.0001,verbosec=0,encoding='utf-8'):
			"""
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
			"""
			self.lr=lr
			self.lr_update_rate=lru
			self.dim=dim
			self.ws=ws
			self.epoch=epoch
			self.min_count=minc
			self.neg=neg
			self.word_ngrams=ngram
			self.loss=loss
			self.bucket=bucket
			self.minn=minn
			self.maxn=maxn
			self.n_thread=th
			self.samplet=t
			self.silent=verbosec
			self.enc=encoding
	def fit(self,X,modelname='model'):
		pass
	def getproperties(self):
                '''
                Input: Nothing, other than object self pointer
                Return: None , prints the descriptions of the model hyperparameters
                '''
                print("The model has following hyperparameters as part of its specification")
                print("Learning rate :"+ str(lr))
                print("Learning rate update after "+str(self.lr_update_rate)+"iterations")
                print("Embedding size: "+str(self.dim))
                print("Epochs :"+ str(self.epochs)
                print("minimal number of word occurences: "+self.min_count)
                print("number of negatives sampled :"+str(self.neg))
                print("max length of word ngram "+str(self.word_ngrams))
                print("loss function: "+str(self.loss))
                print("number of buckets "+str(self.bucket))
                print("min length of char ngram:"+str(self.minn))
                print("min length of char ngram"+ str(self.maxn))
                print("number of threads: "+str(self.n_thread))
                print("sampling threshold"+str(self.samplet))
                print("Verbose log output from the C++ extension enable=1/disble=0:"+ str(self.silent))
                print("input_file encoding :"+str(self.enc))
                return None
            
	def getwords(self):
		'return word list'
		pass
	def getvector(self):
		'return embedding'
		pass

class cbowFastText((BaseEstimator,ClassifierMixin):
	def __init__(self,lpr='__label__',lr=0.1,lru=100,dim=100,ws=5,epoch=5,minc=1,neg=5,ngram=1,\
loss='softmax',nbucket=0,minn=0,maxn=0,th=12,t=0.0001,verbosec=0,encoding='utf-8'):			"""
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

			"""
			self.lr=lr
			self.lr_update_rate=lru
			self.dim=dim
			self.ws=ws
			self.epoch=epoch
			self.min_count=minc
			self.neg=neg
			self.word_ngrams=ngram
			self.loss=loss
			self.bucket=bucket
			self.minn=minn
			self.maxn=maxn
			self.n_thread=th
			self.samplet=t
			self.silent=verbosec
			self.enc=encoding	
	def fit(self,X,modelname='model'):
		pass	
	def getproperties(self):
		'''
                Input: Nothing, other than object self pointer
                Return: None , prints the descriptions of the model hyperparameters
                '''
                print("The model has following hyperparameters as part of its specification")
                print("Learning rate :"+ str(lr))
                print("Learning rate update after "+str(self.lr_update_rate)+"iterations")
                print("Embedding size: "+str(self.dim))
                print("Epochs :"+ str(self.epochs)
                print("minimal number of word occurences: "+self.min_count)
                print("number of negatives sampled :"+str(self.neg))
                print("max length of word ngram "+str(self.word_ngrams))
                print("loss function: "+str(self.loss))
                print("number of buckets "+str(self.bucket))
                print("min length of char ngram:"+str(self.minn))
                print("min length of char ngram"+ str(self.maxn))
                print("number of threads: "+str(self.n_thread))
                print("sampling threshold"+str(self.samplet))
                print("Verbose log output from the C++ extension enable=1/disble=0:"+ str(self.silent))
                print("input_file encoding :"+str(self.enc))
                
	def getwords(self):
		'return word list'
		pass
	def getvector(self):
		'return embedding'
		pass

		

