"""

Data class requires the following parameters:
 - .csv path OR data array (with feature names)
 - target column (by default last one)
 - non-actionable feature columns (index list)
 - categorical feautre columns (index list)


Included example datasets: 
 - "diabetes": diabetes dataset
 - "grad": graduate admissions dataset

"""


# -- Import list -- 
import pandas as pd
import numpy as np


class Data:

	def __init__(self, path = '', data = None, target = -1, exception = [], categorical = [], example = ''):

		if (example != ''):
			# -- Available example datasets -- 
			if (example == "diabetes"):
				df = pd.read_csv("sample_data/diabetes.csv")

			elif (example == "grad"):
				df = pd.read_csv("sample_data/admissions.csv")

			else: 
				raise ValueError("Unknown example dataset chosen")


			self.feature_names = np.array(df.columns)[:-1]
			all_data = np.array(df.values)



		else:
			if ((data is None) & (path == '') & (example == '')):
				raise ValueError("Should provide either a data array or the path to the data")

			elif (data is None):
				df = pd.read_csv(path)
				self.feature_names = np.array(df.columns)[:-1]
				all_data = np.array(df.values)


			else:
				all_data = np.array(data)
				self.feature_names = all_data[0]  # Feature names should be first row


		# -- Split data and target values --
		self.target = all_data[:,target]
		self.data = np.delete(all_data, target, 1)
		self.no_samples, self.no_features = self.data.shape

		# -- Specifying exceptions & categoricals --
		self.ex = exception
		self.cat = categorical





