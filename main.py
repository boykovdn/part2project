import pandas as pd
import numpy as np


class Loader:
	def __init__(self):
		""" This class loads raw data and slices it in ways so as to be passed to the 
		    rest of the program.  """
		self.dataframe = pd.DataFrame.from_csv("./data/2012_S1W2.csv")  # Change if data not locally available
	
	def get_linsolver_coef_raw_dataframe(self):
		""" Picks subset of data to be passed to LinSolver. """
		dataframe_result = pd.DataFrame()
		dataframe_result["az"] = self.dataframe["az"]
		dataframe_result["el"] = self.dataframe["el"]
		dataframe_result["delay1"] = self.dataframe["delay1"]
		dataframe_result["delay2"] = self.dataframe["delay2"]
		dataframe_result.reset_index(drop=True, inplace=True)

		return dataframe_result


class LinSolver:

	def __init__(self):
		""" This class handles the solution of a large set of linear equations, and
	    	    and provides functionality for error analysis.  """
		pass

	def parse_matrix_coef_from_data(self, dataframe):
		""" Calculates coefficients of linear equations from slice of original data.  """ 
		#TODO Calculate indices

	#TODO Write matrix inversion script, get some initial results
	


def main():
	l = Loader()
	print(l.get_linsolver_coef_raw_dataframe().head())


if __name__ == "__main__":
	main()
