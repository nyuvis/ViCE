"""

Model class requires the following parameters:
 - model element or model path OR target column (by default last one)
 - model framework
 	> scikit
 	> tf (coming soon)
 	> pytorch (coming soon)

"""


# -- Import list -- 
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# import data
import pickle



class Model:


	def __init__(self, model=None, backend='scikit'):
		if (backend == 'scikit'):
			self.__class__  = ModelScikit
		
		# tf and pytorch implementation coming soon
		elif (backend == 'tf'):
			self.__class__  = ModelTF

		elif (backend == 'pytorch'):
			self.__class__  = ModelPytorch

		
		self.__init__(model, backend)




class ModelScikit:

	def __init__(self, model=None, backend='scikit'):
		self.model = model
		self.backend = backend


	def train_model(self, data, type = 'svm'):
		"""
		Possible model types:
		- 'svm': support vector machine
		- 'lg': logistic regression
		- 'rf': random forest classifier 

		"""

		rand_state = 0
		# X = data.data
		# y = data.target
		X = data.X
		y = data.y

		# -- Needs to be retained for inserting new samples -- # Incomplete
		# scaler = StandardScaler()
		# X = scaler.fit_transform(data.data)
		# self.mean = scaler.mean_
		# self.scale = scaler.scale_

		# -- Data split -- 
		X_tr, X_test, y_tr, y_test = train_test_split(X,y, test_size=0.2, random_state=rand_state)


		if (type == 'svm'):
			c_val = 1
			self.model = svm.SVC(kernel='linear', C=c_val, probability=True, random_state=rand_state)
			self.model.fit(X_tr,y_tr.reshape(y_tr.shape[0],))

		elif (type == 'rf'):
			self.model = RandomForestClassifier(n_estimators=100, random_state=self.rand_state)
			self.model.fit(X_tr,y_tr.reshape(y_tr.shape[0],))

		elif (type == 'lg'):
			self.model = LogisticRegression(random_state=self.rand_state)
			self.model.fit(X_tr,y_tr.reshape(y_tr.shape[0],))

		else:
			raise ModelError("Unknown model type selected")
		return self


	def save_model(self, out_path):
		if (model is not None):
			pickle.dump(model, open(out_path, 'wb'))

		else:
			raise ModelError("Model incomplete")


	def load_model(self, model_path):
		if model_path != '':
			self.model = pickle.load(open(model_path, 'rb'))
		else:
			raise ModelError("No model path provided")


	def run_model(self, sample):
		if (not self.model):
			raise ModelError("Model incomplete")

		result = self.model.predict_proba(sample.reshape(1, -1))
		return result[0][1]


	def run_model_data(self, data_set):
		pass

	def model_complete(self): 
		return (model != None)

	def model_performance(self):
		print("Test Accuracy: ")
		print("Training Accuracy: ")





class ModelTF:

	def __init__(self, model=None, backend='scikit'):
		self.model = model
		self.backend = backend

	def train_model(self, data, type = 'svm'):
		pass

	def save_model(self, out_path):
		pass


	def load_model(self, model_path):
		pass

	def run_model(self, sample):
		pass

	def run_model_data(self, data_set):
		pass

	def model_complete(self): 
		pass

	def model_performance(self):
		pass

class ModelPytorch:

	def __init__(self, model=None, backend='scikit'):
		self.model = model
		self.backend = backend

	def train_model(self, data, type = 'svm'):
		pass

	def save_model(self, out_path):
		pass


	def load_model(self, model_path):
		pass

	def run_model(self, sample):
		pass

	def run_model_data(self, data_set):
		pass

	def model_complete(self): 
		pass

	def model_performance(self):
		pass
		
	

