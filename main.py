import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Loader:
    def __init__(self, filename = "2012_S1W2.csv"):
        """ This class loads raw data and slices/modifies it in ways so as to be passed to the 
            rest of the program.  """
        self.dataframe = pd.DataFrame.from_csv("./data/" + filename)  # Change if data not locally available
    
    def get_linsolver_coef_raw_dataframe(self, dataframe):
        """ Picks subset of data to be passed to LinSolver. """
        dataframe_result = pd.DataFrame()
        dataframe_result["az"] = dataframe["az"]
        dataframe_result["el"] = dataframe["el"]
        dataframe_result["delay1"] = dataframe["delay1"]
        dataframe_result["delay2"] = dataframe["delay2"]
        dataframe_result.reset_index(drop=True, inplace=True)

        return dataframe_result

    def day_index_dataframe(self):
        """ Adds another column that indicates which day of measurements 
            the data relates to.
            
            Performance of this function is not very important, because it
            is only used once for utility purposes.
        """

        #days_array = np.zeros((self.dataframe.size, 1), dtype=int)        
        days_array = [0]*self.dataframe["h"].size

        current_day = 0

        for i in range(0, self.dataframe["h"].size - 1):
            if self.dataframe["h"].iloc[i+1] < self.dataframe["h"].iloc[i]:
                days_array[i] = current_day
                current_day += 1
            else:
                days_array[i] = current_day

        days_array[-1] = days_array[-2]

        self.dataframe["day_number"] = days_array

class LinSolver:

    def __init__(self):
        """ This class handles the solution of a large set of linear equations, and
                and provides functionality for error analysis.  """
        pass

    def parse_matrix_coef_from_data(self, dataframe):
        """ Calculates the coefficients matrix of linear equations from slice of original data.  """ 
        coefs_list = []  # List so that it can be dynamically extended. TODO implement more efficient data structure      

        """
        The following procedure could be implemented in a more efficient
        way, without using Python lists. I went with this approach because
        lists can be dynamically extended as opposed to arrays, which is
        convenient. 
        
        I implement the formula:

            cos(el)cos(az)*x + cos(el)sin(az)*y + sin(el)*z + A = d2 - d1

        as derived from the geometry of the problem.

        A is an unknown constant offset in the delays, as discussed in the
        manual. I fit for it as well, giving it a constant coefficient of 1
        for every data point.
        """
        
        # The following lambda function returns [coef_x, coef_y, coef_z, 1]
        add_coefs_to_local_array = lambda series : coefs_list.append(
            [np.cos(np.radians(series["el"])) * np.sin(np.radians(series["az"])),
             np.cos(np.radians(series["el"])) * np.cos(np.radians(series["az"])),
             np.sin(np.radians(series["el"])),
             1]
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

    #TODO x-y signs?
    #TODO Extract uncertainties in positions from residual
    

def main():
    l = Loader("2012_all.csv")
    ls = LinSolver()

    l.day_index_dataframe()
    subset = l.dataframe.loc["D/S1E1P51B46"]

    A = ls.parse_matrix_coef_from_data(l.get_linsolver_coef_raw_dataframe(subset))
    b = ls.parse_delays_vector_from_data(subset)
    print(ls.solve_linear_system(A,b))

    
if __name__ == "__main__":
    main()

    """ Tests
    i_vals = []
    residuals = []
    for i in range(10,700):
        dataframe = l.get_linsolver_coef_raw_dataframe()[:i]
        A = ls.parse_matrix_coef_from_data(dataframe)
        b = ls.parse_delays_vector_from_data(dataframe)
        i_vals.append(i)
        residuals.append(ls.solve_linear_system(A,b)[1][0])

    plt.plot(i_vals, residuals)
    plt.title("Residual vs data points used data[:x]")
    plt.xlabel("number of data points")
    plt.ylabel("residual")
    plt.show()
    """

    """Test solving for a pair
    df = l.get_linsolver_coef_raw_dataframe(l.dataframe)  #TODO Test again for sure
    A = ls.parse_matrix_coef_from_data(df)
    b = ls.parse_delays_vector_from_data(df)
    """ 

    """ Test show different POP settings can be fitted with same A
    l = Loader("2012_all.csv")
    ls = LinSolver()

    l.day_index_dataframe()
    subset = l.dataframe.loc["D/S1W2P15B21"]

    A = ls.parse_matrix_coef_from_data(l.get_linsolver_coef_raw_dataframe(subset))
    b = ls.parse_delays_vector_from_data(subset)
    print(ls.solve_linear_system(A,b))

    subset = l.dataframe.loc["D/S1W2P45B13"]

    A = ls.parse_matrix_coef_from_data(l.get_linsolver_coef_raw_dataframe(subset))
    b = ls.parse_delays_vector_from_data(subset)
    print(ls.solve_linear_system(A,b))
    """

    """ Test 1: show different POP settings for S1W2
    l = Loader("2012_all.csv")
    ls = LinSolver()

    l.day_index_dataframe()
    subset = l.dataframe.loc["D/S1W2P15B21"]

    A = ls.parse_matrix_coef_from_data(l.get_linsolver_coef_raw_dataframe(subset))
    b = ls.parse_delays_vector_from_data(subset)
    print(ls.solve_linear_system(A,b))

    subset = l.dataframe.loc["D/S1W2P45B13"]

    A = ls.parse_matrix_coef_from_data(l.get_linsolver_coef_raw_dataframe(subset))
    b = ls.parse_delays_vector_from_data(subset)
    print(ls.solve_linear_system(A,b))

    subset = l.dataframe.loc["D/S1W2P15B32"]

    A = ls.parse_matrix_coef_from_data(l.get_linsolver_coef_raw_dataframe(subset))
    b = ls.parse_delays_vector_from_data(subset)
    print(ls.solve_linear_system(A,b))
    print(subset)
    """
