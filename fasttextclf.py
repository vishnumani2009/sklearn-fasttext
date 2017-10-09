from sklearn.base import BaseEstimator, ClassifierMixin
import fasttext as ft
from sklearn.metrics import classification_report

class FastTextClassifier(BaseEstimator,ClassifierMixin):
"""Base classiifer of Fasttext estimator"""

	def __init__(self,lpr='__label__',lr=0.1,lru=100,dim=100,ws=5,epoch=5,minc=1,neg=5,ngram=1,loss='softmax',nbucket=0,minn=0,maxn=0,thread=2,silent=1,output="model"):
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
		todo : Recheck need of some of the variables, present in default classifier
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
		self.thread=thread
		self.silent=silent
		self.classifier=None
		self.result=None

		self.output=output

	def fit(self,input_file):
                '''
                Input: takes input file in format
                returns classifier object
                to do: add option to feed list of X and Y or file
                '''
                
                self.classifier = ft.supervised(input_file, self.output, dim=self.dim, lr=self.lr, epoch=self.epoch, min_count=self.min_count, word_ngrams=self.word_ngrams, bucket=self.bucket, thread=self.thread, silent=self.silent, label_prefix=self.lpr)
                return(self.classisifer)
            
	def predict(self,test_file,csvflag=True,reports=False):
                '''
                Input: Takes input test finle in format
                return results object
                to do: add unit tests using sentiment analysis dataset 
                to do: Add K best labels options for csvflag = False 
                to do: add report option
                '''
                try:
                    if type(test_file) == 'list' and csvflag=False:
                        self.result=self.classifier.predict(test_file)
                    else:
                        print "Error in input"
                    if csvflag:
                            self.result=self.classifier.test(test_file)
                except:
                    print("Exception in predict call error in format of test_file/input sentence list")
                return(self.result)
                
	def report(self,ytrue,ypred):
                '''
                Input: predicted and true labels
                return reort of classification
                to do: add label option and unit testing
                
                '''
                print(classification_report(ytrue,ypred))
                return None
            
	def predict_proba(self,X):
                '''
                Input: List of sentences
                return reort of classification
                to do: check output of classifier predct_proba add label option and unit testing
                '''
                labels=self.classifier.predict_proba(X)
                return(labels)

	def getlabels(self):
                '''
                Input: None
                returns: Class labels in dataset
                to do : check need of the this funcion
                '''
		return(self.classifier.labels)
		
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
			self.enc=encodings
			self.model=None
			self.result=None
			

            
	def fit(self,X,modelname='model',csvflag=False):
                '''
                Input: takes input file in format
                returns classifier object
                to do: add option to feed list of X and Y or file
                to do: check options for the api call 
                to do: write unit test
                '''
                try:
                    if not csvflag:
                        self.model=ft.skipgram(X, modelname, lr=self.lr, dim=self.dim,lr_update_rate=self.lr_update_rate,epoch=self.epoch,bucket=self.bucket,loss=self.loss,thread=self.n_thread)
                except:
                    print("Error in input dataset format")
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
                """to do: check words list"""
		return(self.model.words)
            # list of words in dictionary)
	def getvector(self,word=None):
                """
                to do : add try catch for word type
                to do: add try catch for word existance 
                """
                return(self.model[word])
            
            
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
                '''
                Input: takes input file in format
                returns classifier object
                to do: add option to feed list of X and Y or file
                to do: check options for the api call 
                to do: write unit test
                '''
		try:
                    if not csvflag:
                        self.model=ft.cbow(X, modelname, lr=self.lr, dim=self.dim,lr_update_rate=self.lr_update_rate,epoch=self.epoch,bucket=self.bucket,loss=self.loss,thread=self.n_thread)
                except:
                    print("Error in input dataset format")
                    
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
		"""to do: check words list"""
		return(self.model.words)

	def getvector(self):
		       """
                to do : add try catch for word type
                to do: add try catch for word existance 
                """
                return(self.model[word])

		

