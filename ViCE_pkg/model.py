"""

Model class requires the following parameters:
 - model element or model path OR target column (by default last one)
 - model framework
 	> scikit
 	> tf (in development)
 	> pytorch (in development)

"""


# -- Import list -- 
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import data



class Model:


	def __init__(self, model=None, model_path='', backend='scikit'):

		# if((model is None) & (model_path == '')):
		# 	raise ValueError("should provide either a trained model or the path to a model")

		# else:

		if (backend == 'scikit'):
			self.__class__  = ModelScikit
		
		# tf and pytorch implementation coming soon

		self.__init__(model, model_path, backend)




class ModelScikit:

	def __init__(self, model=None, model_path='', backend='scikit'):
		self.model = model
		self.model_path = model_path
		self.backend = backend


	def train_model(self, data, type = 'svm'):
		"""
		Possible model types:
		- 'svm': support vector machine
		- 'lg': logistic regression
		- 'rf': random forest classifier 

		"""

		rand_state = 0
		X = data.data
		y = data.target

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


	def save_model(self, out_path):
		if (model is not None):
			pickle.dump(model, open(out_path, 'wb'))

		else:
			raise ModelError("Model incomplete")


	def load_model(self):
		if self.model_path != '':
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
		# if (not self.model):
		# 	raise ModelError("Train Model First")

		# for i in range(data_set.shape[0]):
		# 	data_set[i] = self.__scaled_row(data_set[i])

		# pred = self.model.predict(data_set)
		# self.model_calls += data_set.shape[0]
		# return pred


	def model_complete(self): 
		return (model != None)

	def model_performance(self):
		print("Test Accuracy: ")
		print("Training Accuracy: ")

		
	

