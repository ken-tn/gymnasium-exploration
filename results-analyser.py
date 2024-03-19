import pandas as pd
import matplotlib.pyplot as plt
import pickle

experimentName = "combined_heuristic_convolution_modelfix"
with open("results/{}.pkl".format(experimentName), 'rb') as resultsFile:
    results = pickle.load(resultsFile)
    
    df = pd.DataFrame(results)

    

    # df_days_calories = pd.DataFrame( 
    #     {'Games' : episodes ,  
    #     'Reward / 100': rewards ,  
    #     'Drawn Pieces / 10': pieces,
    #     'Lines Cleared': linesCleared,
    #     'Score': score,
    #     'Epsilon': epsilon}) 
    
    # ax = plt.gca()
    
    # #use plot() method on the dataframe 
    # df_days_calories.plot( x = 'Games' , y = 'Lines Cleared' , ax = ax ) 
    # df_days_calories.plot( x = 'Games' , y = 'Epsilon' , ax = ax ) 
    # #df_days_calories.plot( x = 'Games' , y = 'Score' , ax = ax ) 
    # plt.show()