"""

Data class requires the following parameters:
 - .csv path
 - target column (by default last one)
 - non-actionable feature columns (index list)
 - categorical feautre columns (index list)

"""


# -- Import list -- 
import pandas as pd
import numpy as np


class Data:

	def __init__(self, path, target = -1, exception = [], categorical = []):

		df = pd.read_csv(path)
		all_data = np.array(df.values)
		
		self.feature_names = np.array(df.columns)[:-1]


		# -- Split data and target values --
		self.target = all_data[:,target]
		self.data = np.delete(all_data, target, 1)
		self.no_samples, self.no_features = self.data.shape

		# -- Specifying exceptions & categoricals --
		self.ex = exception
		self.cat = categorical 


		