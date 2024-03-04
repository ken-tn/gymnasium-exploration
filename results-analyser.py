import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open("results.pkl", 'rb') as resultsFile:
    results = pickle.load(resultsFile)
    
    timestamps = [item[0] for item in results]
    episodes = [item[1] for item in results]
    rewards = [item[2] for item in results]
    epsilon = [item[3] for item in results]
    pieces = [item[4] for item in results]
    linesCleared = [item[5] for item in results]
    tetrisClears = [item[6] for item in results]
    

    df_days_calories = pd.DataFrame( 
        {'Games' : episodes ,  
        'Reward': rewards ,  
        'Pieces': pieces,
        'Lines Cleared': linesCleared}) 
    
    ax = plt.gca()
    
    #use plot() method on the dataframe 
    df_days_calories.plot( x = 'Games' , y = 'Reward', ax = ax ) 
    df_days_calories.plot( x = 'Games' , y = 'Pieces' , ax = ax ) 
    df_days_calories.plot( x = 'Games' , y = 'Lines Cleared' , ax = ax ) 
    plt.show()