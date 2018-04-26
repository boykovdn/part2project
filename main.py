import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class Loader:
    def __init__(self, filename = "2012_S1W2.csv"):
        """ This class loads raw data and slices/modifies it in ways so as to be passed to the 
            rest of the program.  """
        self.dataframe = pd.DataFrame.from_csv("./data/" + filename)  # Change if data not locally available
        self.day_index_dataframe()  # Add the day index on loading
    
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

    
    @staticmethod
    def get_telescope_pop_pairs(dataframe):
        """ INPUT: A dataframe indexed by telescope-pop identifiers
            OUTPUT: A list of the unique telescope-pop identifiers

            Output from this function is used for grouping the data 
            into clusters that have the same constant offset in delay
            measurements. 

            Not enhanced for performance because only serves a utility
            purpose.
        """
        ids = []

        for name in dataframe.index:
            if name not in ids:
                ids.append(name)

        return ids

    def get_telescope_pairs(self, dataframe):
        """This function extracts the unique telescope pairs
        """
        
        pairs = []
    
        for identifier in dataframe.index:
            pair = identifier[2:-6]
            if pair not in pairs: 
                pairs.append(pair) 

        return pairs


    def save_persistent_data(self, dataframe):
        """
        This function is used once to save central
        results that persist between subsequent tests.
        """

        dataframe.to_csv(path_or_buf = "results.csv", index = False)

    def load_persistent_data(self):
        """
        This supporting function loads locally saved
        processed data to speed up subsequent processing.
        """

        return pd.read_csv("results.csv")

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

    #TODO Extract uncertainties in positions from residual


class Routines:
    """
    Wrapper class for the numerical routines and results.
    """

    def get_subset_constant_offset(self, dataframe):
        """
        Routine for grouping the raw data into subsets of
        constant delay offset
    
        Data has to be grouped once into subsets of same
        telescope-telescope and POP configuration pairs,
        and then these subsets are split into different
        data for different days. This ensures that the
        regression analysis is fitting data that corres-
        ponds to costant offsets as POP delay varies on day.
        """
        pairs = Loader.get_telescope_pop_pairs(dataframe)
    
        subsubsets = []
        for pair in pairs:
            subset = dataframe.loc[[pair]]
            unique_days = []
            for day in subset["day_number"]:
                if day not in unique_days:
                    unique_days.append(day)
                    subsubsets.append(subset.loc[subset["day_number"] == day])
    
        return subsubsets

    def filter_underconstrained(self, subsets): #  O(n) in subset number
                
        results = []
        for s in subsets:
            if s.index.size > 4:
                results.append(s)

        return results 


    def solution_routine(self, subsets):
        l = Loader("2012_all.csv")
        ls = LinSolver()
    
        xs = []
        ys = []
        zs = []
        const = []
        res = []
        pairs = []
        data_number = []
        day_number = []

        subsets = self.filter_underconstrained(subsets)  # O(n) in subset number

        for s in subsets:
            pairs.append((s.index)[0][2:-6])
            A = ls.parse_matrix_coef_from_data(s)
            b = ls.parse_delays_vector_from_data(s)
            solution = ls.solve_linear_system(A,b)
            xs.append(solution[0][0])
            ys.append(solution[0][1])
            zs.append(solution[0][2])
            const.append(solution[0][3])
            res.append(solution[1])
            data_number.append(s.index.size)
            day_number.append(s["day_number"].iloc[0])
    
        results = pd.DataFrame({"day_number": day_number, "pair": pairs,"x": xs,"y": ys,"z": zs,"offset": const,"residue": res, "data_number": data_number})
    
        results = results[["pair","x","y","z","offset","residue","data_number","day_number"]]
        results["baseline"] = np.sqrt(np.power(results["x"],2) + np.power(results["y"],2) + np.power(results["z"],2))
    
        return results
    
    def get_coordinates(self, dataframe):
        """
        INPUT: Dataframe of solutions for different
        constant delays offsets, but a single telescope
        pair

        OUTPUT: A single solution for the telescope pair
        """        

        #Exclude underdefined solutions (with less than 4 datapoints)
        dataframe = dataframe.drop(dataframe.loc[dataframe["data_number"] < 4].index)
        
        pair = "underconstrained"
        x = 0
        y = 0
        z = 0

        x_std = 0
        y_std = 0
        z_std = 0

        data_number = 0

        if dataframe.index.size == 0:
            pass
        else:
            pair = dataframe["pair"].iloc[0]
            x = np.average(dataframe["x"])
            y = np.average(dataframe["y"])
            z = np.average(dataframe["z"])
            x_std = np.std(dataframe["x"])
            y_std = np.std(dataframe["y"])
            z_std = np.std(dataframe["z"])
            data_number = dataframe.index.size

        return [pair,x,y,z,x_std,y_std,z_std,data_number]

    def remove_outliers(self, dataframe):
        """
        INPUT: Dataframe of pair,x,y,z values
        
        OUTPUT: Dataframe of pair,x,y,z values with
        outliers in x,y,z removed
        """

        current = dataframe["x"].iloc[0]
        flagged_indices = []
        for i in range(0, dataframe["x"].size):
            if current - dataframe["x"].iloc[i] > 1:
                flagged_indices.append(dataframe.index[i])

        dataframe.drop(labels=flagged_indices, inplace=True) 
        
        current = dataframe["y"].iloc[0]
        flagged_indices = []
        for i in range(0, dataframe["y"].size):
            if current - dataframe["y"].iloc[i] > 1:
                flagged_indices.append(dataframe.index[i])

        dataframe.drop(labels=flagged_indices, inplace=True) 
        
        current = dataframe["z"].iloc[0]
        flagged_indices = []
        for i in range(0, dataframe["z"].size):
            if current - dataframe["z"].iloc[i] > 1:
                flagged_indices.append(dataframe.index[i])

        dataframe.drop(labels=flagged_indices, inplace=True) 
        
        return dataframe


    def remove_outliers_coord(self, dataframe, coordinate_name):
        """
        INPUT: dataframe of pair, x, y, z values; "x", "y", or "z"
        OUTPUT: dataframe with outliers in this particular coordinate
        removed

        If the dataframe is empty, return it.
        """

        if(dataframe.index.size == 0):
            print("returned empty dataframe..")
            return dataframe

        else:
            bins = np.arange(-330,330)       
     
            hist, hist_edges = np.histogram(dataframe[coordinate_name], bins=bins)
            value_lower = hist_edges[np.argmax(hist)]
            subset = dataframe.loc[dataframe[coordinate_name] > value_lower]
            subset = subset.loc[subset[coordinate_name] < (value_lower + 1)]
    
            average = np.average(subset[coordinate_name])
            stdev = np.std(subset[coordinate_name])
            
            subset = subset.loc[subset[coordinate_name] > (average - 3 * stdev)]
            subset = subset.loc[subset[coordinate_name] < (average + 3 * stdev)]
       
            return subset



    def remove_outliers_1(self, dataframe):
        """
        INPUT: Dataframe of pair, x, y, z values
        OUTPUT: Dataframe of pair,x,y,z, values and
        outliers in x,y,z removed.

        This is a different implementation of remove_outliers. 

        Here, a histogram is used to detect in what range the 
        answer lies. In order to avoid crossover between bins 
        in cases where the value is close to the bin edge, the
        algorithm computes the average value of the bin and
        runs through the entire dataset, thresholding the values
        to only those that lie within +- 1 of this value, which
        is much higher than the sigma of the value distribution.
        """

        for coord in ["x", "y", "z"]:
            dataframe = self.remove_outliers_coord(dataframe, coord)

        return dataframe


    def solution_routine_s1w2(self, dataframe):
        l = Loader()
        ls = LinSolver()
    
        xs = []
        ys = []
        zs = []
        res = []
        day_number = []
        data_number = []

        for day in range(dataframe["day_number"].iloc[dataframe.index.size - 1]):
            s = dataframe.loc[dataframe["day_number"] == day]
            A = ls.parse_matrix_coef_from_data(s)
            b = ls.parse_delays_vector_from_data(s)
            solution = ls.solve_linear_system(A,b)
            xs.append(solution[0][0])
            ys.append(solution[0][1])
            zs.append(solution[0][2])
            res.append(solution[1])
            day_number.append(day)
            data_number.append(s.index.size)
    
        results = pd.DataFrame({"x": xs,"y": ys,"z": zs, "residue": res, "data_number": data_number, "day_number": day_number}) 
    
        results = results[["x","y","z","day_number", "residue","data_number"]]
        results["baseline"] = np.sqrt(np.power(results["x"],2) + np.power(results["y"],2) + np.power(results["z"],2))
    
        return results

    def get_seizmic_avg_std(self):
        l = Loader("2012_all.csv")
        ls = LinSolver()
        r = Routines()
    
        pairs = l.get_telescope_pairs(l.dataframe)    
    
        subsets = r.get_subset_constant_offset()
        results = r.solution_routine(subsets)
    
        gradients = [] 
        for pair in pairs:
            results_pair = results.loc[results["pair"] == pair]
            results_pair = r.remove_outliers_1(results_pair)
            for coord in ["x","y","z"]:
                coefs = np.polyfit(results_pair["day_number"], results_pair[coord], 1)
                gradients.append(coefs[0]) 

        results_pair = results.loc[results["pair"] == "E1W2"]
        results_pair = r.remove_outliers_1(results_pair) 
        
        return np.average(gradients), np.std(gradients)
    

def main():

if __name__ == "__main__":
    main()

    """ Test: timing execution
    data_numbers = []
    times = []
    for i in np.linspace(10, 23000, 13):

        i = int(i)
        t1 = time.time()

        l = Loader("2012_all.csv")
        l.dataframe = l.dataframe.iloc[:i]
    
        ls = LinSolver()
        r = Routines()
        pairs = l.get_telescope_pairs(l.dataframe)    
    
        subsets = r.get_subset_constant_offset(l.dataframe)
        results = r.solution_routine(subsets)
    
        pairs_df = []
        xs = [] 
        ys = []
        zs = []
        xs_std = []
        ys_std = []
        zs_std = []
        data_number = []
       
        for pair in pairs:  
            dataframe = results.loc[results["pair"] == pair]  # Select data
            dataframe = r.remove_outliers_1(dataframe)
            coords = r.get_coordinates(dataframe)
            pairs_df.append(coords[0])
            xs.append(coords[1])
            ys.append(coords[2])
            zs.append(coords[3])
            xs_std.append(coords[4])
            ys_std.append(coords[5])
            zs_std.append(coords[6])
            data_number.append(coords[7])
     
        result = pd.DataFrame({"pairs":pairs_df, "x":xs, "y":ys, "z":zs, "x_std": xs_std, "y_std": ys_std, "z_std": zs_std, "data_number":data_number})    
        result = result[["pairs","x","y","z","x_std","y_std","z_std","data_number"]]
        result.round(decimals=2)
       
        t2 = time.time()
    
        data_numbers.append(i)
        times.append(t2-t1)

    plt.scatter(data_numbers, times)
    plt.xlabel("datapoints, total")
    plt.ylabel("execution time / s")

    plt.show()

    """


    """ Final results
    l = Loader("2012_all.csv")
    ls = LinSolver()
    r = Routines()

    pairs = l.get_telescope_pairs(l.dataframe)    

    subsets = r.get_subset_constant_offset(l.dataframe)
    results = r.solution_routine(subsets)

    
    pairs_df = []
    xs = [] 
    ys = []
    zs = []
    xs_std = []
    ys_std = []
    zs_std = []
    data_number = []

    for pair in pairs:  
        dataframe = results.loc[results["pair"] == pair]  # Select data
        dataframe = r.remove_outliers_1(dataframe)
        coords = r.get_coordinates(dataframe)
        pairs_df.append(coords[0])
        xs.append(coords[1])
        ys.append(coords[2])
        zs.append(coords[3])
        xs_std.append(coords[4])
        ys_std.append(coords[5])
        zs_std.append(coords[6])
        data_number.append(coords[7])
 
    result = pd.DataFrame({"pairs":pairs_df, "x":xs, "y":ys, "z":zs, "x_std": xs_std, "y_std": ys_std, "z_std": zs_std, "data_number":data_number})    
    result = result[["pairs","x","y","z","x_std","y_std","z_std","data_number"]]
    result.round(decimals=2)
    result.to_csv(path_or_buf = "final_results.csv", index=False)
    
    print(result)


    """
    """Tests S1W2 residual buildup
    l = Loader()
    ls = LinSolver()

    i_vals = []
    residuals = []
    for i in range(10,700):
        dataframe = l.get_linsolver_coef_raw_dataframe(l.dataframe)[:i]
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

    """ Test 1: show different POP settings for S1W2. There seems to be an outlier.
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

    """Test2 Solve for specific days
    l = Loader("2012_all.csv")
    ls = LinSolver()

    l.day_index_dataframe()
    df = l.dataframe
    subset_all = df.loc["D/W1W2P25B31"]

    subset_day34 = subset_all.loc[subset_all["day_number"] == 34]
    subset_day144 = subset_all.loc[subset_all["day_number"] == 144]
    subset_day143 = subset_all.loc[subset_all["day_number"] == 143]

    print(subset_all.size)
    print(subset_day34.size)
    print(subset_day144.size)
    print(subset_day143.size)

    A = ls.parse_matrix_coef_from_data(l.get_linsolver_coef_raw_dataframe(subset_all))
    b = ls.parse_delays_vector_from_data(subset_all)
    print(ls.solve_linear_system(A,b))
    
    A = ls.parse_matrix_coef_from_data(l.get_linsolver_coef_raw_dataframe(subset_day34))
    b = ls.parse_delays_vector_from_data(subset_day34)
    print(ls.solve_linear_system(A,b))

    A = ls.parse_matrix_coef_from_data(l.get_linsolver_coef_raw_dataframe(subset_day144))
    b = ls.parse_delays_vector_from_data(subset_day144)
    print(ls.solve_linear_system(A,b))

    A = ls.parse_matrix_coef_from_data(l.get_linsolver_coef_raw_dataframe(subset_day143))
    b = ls.parse_delays_vector_from_data(subset_day143)
    print(ls.solve_linear_system(A,b))


    """
    """ Test 22 April filtered solutions
    l = Loader("2012_all.csv")
    ls = LinSolver()
    r = Routines()

    pairs = l.get_telescope_pairs(l.dataframe)    

    subsets = r.get_subset_constant_offset(l.dataframe)
    results = r.solution_routine(subsets)

    for pair in pairs:  
        dataframe = results.loc[results["pair"] == pair]  # Select data
        dataframe = r.remove_outliers_1(dataframe)
        print(r.get_coordinates(dataframe))
    """
    """ Test seismic linear fit
    l = Loader("2012_all.csv")
    ls = LinSolver()
    r = Routines()

    pairs = l.get_telescope_pairs(l.dataframe)    

    subsets = r.get_subset_constant_offset(l.dataframe)
    results = r.solution_routine(subsets)

 
    for pair in pairs:
        results_pair = results.loc[results["pair"] == pair]
        results_pair = r.remove_outliers_1(results_pair)
        for coord in ["x","y","z"]:
            coefs = np.polyfit(results_pair["day_number"], results_pair[coord], 1)
            print(coefs, pair, coord)

    results_pair = results.loc[results["pair"] == "E1W2"]
    results_pair = r.remove_outliers_1(results_pair) 

    plt.scatter(results_pair["day_number"], results_pair["z"])
    plt.show()
    """

    """
    l = Loader("2012_all.csv")
    ls = LinSolver()
    r = Routines()

    pairs = l.get_telescope_pairs(l.dataframe)    
    print(r.get_seizmic_avg_std(), "test")
    subsets = r.get_subset_constant_offset()
    results = r.solution_routine(subsets)

    results_pair = results.loc[results["pair"] == "E1W1"]
    results_pair = r.remove_outliers_1(results_pair) 

    plt.scatter(results_pair["day_number"], results_pair["x"])
    coefs = np.polyfit(results_pair["day_number"], results_pair["x"], 1)
    x_reg = np.arange(150)
    y_reg = x_reg * coefs[0] + coefs[1]
    plt.plot(x_reg, y_reg)
    coefs[0] = np.round(coefs[0], 6)
    coefs[1] = np.round(coefs[1],2)
    plt.text(20,-300.39, str(coefs[0]) + "x" + " " + str(coefs[1]))
    plt.xlabel("Observation day number (exact date unknown)")
    plt.ylabel("x coordinate")
    plt.show()
 
    """
    """
    r = Routines()    
    l = Loader("2012_all.csv")

    pairs = l.get_telescope_pairs(l.dataframe)

    # Save data first, then load to get faster performance on further tests
    # This allows for a computationally heavy operation to be skipped

    #subsets = r.get_subset_constant_offset(l.dataframe)
    #results = r.solution_routine(subsets)

    results = l.load_persistent_data()
    #l.save_persistent_data(results)  # Saves time     

    for pair in pairs:
        pair_dataframe = results.loc[results["pair"] == pair]
        pair_dataframe = r.remove_outliers_1(pair_dataframe)
        print(r.get_coordinates(pair_dataframe))
    """
