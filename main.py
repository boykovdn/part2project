import pandas as pd
import numpy as np


class Loader:
	def __init__(self):
		#TODO Set up dataFrame for each file
		self.dataframe = pd.DataFrame.from_csv("./data/2012_S1W2.csv")  # Change if data not locally available
		


class LinSolver:

	def __init__(self):
		""" This class handles the solution of a large set of linear equations, and
	    	    and provides functionality for error analysis.  """
		pass

	def parse_matrix_coef_from_data(self, dataframe):
		""" Calculates coefficients of linear equations from slice of original data.  """ 
	


def main():
	l = Loader()
	print(l.dataframe.head())


if __name__ == "__main__":
	main()
