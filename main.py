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
        arrayshape = (dataframe.index.size, 3)
        coefs_list = []  # List so that it can be dynamically extended. TODO implement more efficient data structure      

        """
        The following procedure could be implemented in a more efficient
        way, without using Python lists. I went with this approach because
        lists can be dynamically extended as opposed to arrays, which is
        convenient. 
        
        I implement the formula:

            cos(el)cos(az)*x + cos(el)sin(az)*y + sin(el)*z = d2 - d1

        as derived from the geometry of the problem.
        """
        
        # The following lambda function returns [coef_x, coef_y, coef_z]
        add_coefs_to_local_array = lambda series : coefs_list.append(
            [np.cos(np.radians(series["el"])) * np.sin(np.radians(series["az"])),
             np.cos(np.radians(series["el"])) * np.cos(np.radians(series["az"])),
             np.sin(np.radians(series["el"]))]
            )
       
        dataframe.apply(add_coefs_to_local_array, axis=1)

        return np.array(coefs_list)

    def parse_delays_vector_from_data(self, dataframe):
        """ Creates a vector of delay differences to be passed on
        to the solution routine, from the original data.  """

        result_series = dataframe["delay2"] - dataframe["delay1"]
        result_series.reset_index(drop=True, inplace=True)

        return np.array(result_series)

    def solve_linear_system(self, A, b, rcond=None):
        """ This function applies the numpy lstsq routine to the
            parsed data and returns a solution that minimises the
            square error """

        return np.linalg.lstsq(A, b, rcond)

    #TODO Check scaling of data, reiterate to get correct answer
    #TODO Scaling is wrong, along with z value which is 2x larger

def main():
    l = Loader()
    ls = LinSolver()

    dataframe = l.get_linsolver_coef_raw_dataframe()
    A = ls.parse_matrix_coef_from_data(dataframe)
    b = ls.parse_delays_vector_from_data(dataframe)
    print(ls.solve_linear_system(A, b))

if __name__ == "__main__":
    main()
