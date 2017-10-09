from sklearn.base import BaseEstimator, ClassifierMixin
import fasttext as ft
from sklearn.metrics import classification_report

class FastTextClassifier(BaseEstimator,ClassifierMixin):
        """Base classiifer of Fasttext estimator"""

	def __init__(self,lpr='__label__',lr=0.1,lru=100,dim=100,ws=5,epoch=100,minc=1,neg=5,ngram=1,loss='softmax',nbucket=0,minn=0,maxn=0,thread=4,silent=0,output="model"):
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
		self.bucket=nbucket
		self.minn=minn
		self.maxn=maxn
		self.thread=thread
		self.silent=silent
		self.classifier=None
		self.result=None
		self.output=output
		self.lpr=lpr

	def fit(self,input_file):
                '''
                Input: takes input file in format
                returns classifier object
                to do: add option to feed list of X and Y or file
                '''
                self.classifier = ft.supervised(input_file, self.output, dim=self.dim, lr=self.lr, epoch=self.epoch, min_count=self.min_count, word_ngrams=self.word_ngrams, bucket=self.bucket, thread=self.thread, silent=self.silent, label_prefix=self.lpr)
                return(None)
            
	def predict(self,test_file,csvflag=True,k_best=1):
                '''
                Input: Takes input test finle in format
                return results object
                to do: add unit tests using sentiment analysis dataset 
                to do: Add K best labels options for csvflag = False 
                
                '''
                try:
                    if csvflag==False and type(test_file) == 'list':
                        self.result=self.classifier.predict(test_file,k=k_best)
                    if csvflag:
                            lines=open(test_file,"r").readlines()
                            sentences=[line.split(" , ")[1] for line in lines]
                            self.result=self.classifier.predict(sentences,k_best)
                except:
                    print("Error in input dataset.. please see if the file/list of sentences is of correct format")
                    sys.exit(-1)
                self.result=[int(labels[0]) for labels in self.result]
                return(self.result)
                
	def report(self,ytrue,ypred):
                '''
                Input: predicted and true labels
                return reort of classification
                to do: add label option and unit testing
                
                '''
                print(classification_report(ytrue,ypred))
                return None
            
	def predict_proba(self,test_file,csvflag=True,k_best=1):
                '''
                Input: List of sentences
                return reort of classification
                to do: check output of classifier predct_proba add label option and unit testing
                '''
                try:
                    if csvflag==False and type(test_file) == 'list':
                        self.result=self.classifier.predict_proba(test_file,k=k_best)
                    if csvflag:
                            lines=open(test_file,"r").readlines()
                            sentences=[line.split(" , ")[1] for line in lines]
                            self.result=self.classifier.predict_proba(sentences,k_best)
                except:
                    print("Error in input dataset.. please see if the file/list of sentences is of correct format")
                    sys.exit(-1)
                return(self.result)

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
                print("Label prefix used : "+str(self.label_prefix))
                print("Learning rate :"+ str(lr))
                print("Learning rate update after "+str(self.lr_update_rate)+"iterations")
                print("Embedding size: "+str(self.dim))
                print("Epochs :"+ str(self.epochs))
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
		self.classifier=ft.load_model(X,label_prefix=self.lpr)
		