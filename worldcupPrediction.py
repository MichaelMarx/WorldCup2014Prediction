"""
World Cup 2014 Prediction.
04/27/2020
@author: Michael Marx
"""
'''
This file imports data about past soccer results and the team statistics for the 2014 World Cup teams. 
The goal is to fit and train a prediction model in a way that allows us to predict the outcome of the World  Cup.
First, the results of Soccer matches and the statistics of the world cup teams are imported. We will only look 
at games from 2013 and 2014. The teams' stats are added to the results data frame. The differences between the
teams's stats are added, too.
'''
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the teams and stats
teams = pd.read_csv('/home/michael/Documents/SoccerAnalysisProject/Data/teamStats.csv')

# Import all international matches
matches = pd.read_csv('/home/michael/Documents/SoccerAnalysisProject/Data/allresults.csv')

# Clean matches dataframe of unimportant data.
matches = matches.drop(['city', 'tournament', 'country', 'neutral'], axis = 1)
matches.index = pd.DatetimeIndex(matches.date).year
matches = matches.drop('date', 1)

# Only 2013-2014 games and only games between teams that are in the cup (where we have ratings and stats)
matches = matches.loc[2013:2014]
dfTeams = list(teams.team.unique())
matches = matches.reset_index()

# Only games between teams of which we have the stats.
for index, row in matches.iterrows():
    if row.home_team not in dfTeams:
        matches.loc[index, 'home_team'] = None
    if row.away_team not in dfTeams:
        matches.loc[index, 'away_team'] = None
        
matches = matches.dropna()

# Add team stats to matches
matches['homeTeamValue'] = matches.home_team.map(teams.set_index('team')['teamValue'].to_dict())
matches['awayTeamValue'] = matches.away_team.map(teams.set_index('team')['teamValue'].to_dict())
matches['homeTeamGoalsScored'] = matches.home_team.map(teams.set_index('team')['gs'].to_dict())
matches['awayTeamGoalsScored'] = matches.away_team.map(teams.set_index('team')['gs'].to_dict())
matches['homeTeamGoalsConc'] = matches.home_team.map(teams.set_index('team')['gc'].to_dict())
matches['awayTeamGoalsConc'] = matches.away_team.map(teams.set_index('team')['gc'].to_dict())
matches['homeTeamRanking'] = matches.home_team.map(teams.set_index('team')['ranking'].to_dict())
matches['awayTeamRanking'] = matches.away_team.map(teams.set_index('team')['ranking'].to_dict())
matches['homeWins'] = matches.home_team.map(teams.set_index('team')['win'].to_dict())
matches['awayWins'] = matches.away_team.map(teams.set_index('team')['win'].to_dict())
matches['homeDraws'] = matches.home_team.map(teams.set_index('team')['draw'].to_dict())
matches['awayDraws'] = matches.away_team.map(teams.set_index('team')['draw'].to_dict())
matches['homeLoss'] = matches.home_team.map(teams.set_index('team')['loss'].to_dict())
matches['awayLoss'] = matches.away_team.map(teams.set_index('team')['loss'].to_dict())
matches['homePoints'] = matches.home_team.map(teams.set_index('team')['points'].to_dict())
matches['awayPoints'] = matches.away_team.map(teams.set_index('team')['points'].to_dict())

# Adding the game results. 1 for team 1, 0 for draw, -1 for team 2. 
matches['outcome'] = matches.home_score - matches.away_score
matches['winner'] = None
matches['winner'][matches.outcome > 0] = '1'
matches['winner'][matches.outcome < 0] = '-1'
matches['winner'][matches.outcome == 0] = '0'
matches['winner'] = pd.Categorical(matches['winner'])

# Adding the differences of stats to dataframe.
matches['valueDif'] = matches.homeTeamValue - matches.awayTeamValue
matches['GSDif'] = matches.homeTeamGoalsScored - matches.awayTeamGoalsScored
matches['GCDif'] = matches.homeTeamGoalsConc - matches.awayTeamGoalsConc
matches['rankingDif'] = matches.homeTeamRanking - matches.awayTeamRanking
matches['winDif'] = matches.homeWins - matches.awayWins
matches['drawDif'] = matches.homeDraws - matches.awayDraws
matches['lossDif'] = matches.homeLoss - matches.awayLoss
matches['pointsDif'] = matches.homePoints - matches.awayPoints

'''
Next, the data is visualized to see correlations and gain understanding of the data.
'''
# Showing the means and standard deviations of important data.
avgTeamValue = teams['teamValue'].mean()        # Average team value of World Cup teans
print(avgTeamValue)
stdTeamValue = teams['teamValue'].std()         # Standard Deviation of team values
print(stdTeamValue)
avgTeamPoints = teams['points'].mean()          # Average team points during World cup qualification.
print(avgTeamPoints)
stdTeamPoints = teams['points'].std()           # Standard Deviation of teams' points.
print(stdTeamPoints)

# Visualize Data
# Correlation of data.
correlation = matches.corr()
print(correlation)

# Pair plot to see the correlations between data.
pairplot = sns.pairplot(matches)
print(pairplot)

# Scatter plot.
scatter = sns.scatterplot(x = matches.valueDif, y = matches.outcome, palette='Set3', hue=matches.outcome, legend=False)
plt.xticks(rotation=45)
plt.xlabel('Difference of Team Values: +: team 1 more valuabel, -: team 2 more valueable')
plt.ylabel('Outcome: +: team1 wins, -: team 2 wins')
plt.title('Value Difference vs. Outcome')

# Box plot.
box = sns.boxplot(x = matches.rankingDif, y = matches.winner, palette='Set3')
plt.xlabel('Difference of rakings between teams: +: team1 higher ranked, -: team2 higher ranked')
plt.ylabel('Winner: 1: team1 wins, -1: team2 wins, 0: Draw')
plt.title('Difference in rankings vs. Winner')

# Bar plot.
barplot = sns.barplot(x = matches.outcome, y = matches.pointsDif, ci=None)
plt.xlabel('Differene of scored goals in a game: +: team1 won, -: team2 won')
plt.ylabel('Difference in points between teams')
plt.title('Game outcome vs. Points difference')

'''
The graphics showed that in soccer every result is possible. However, the pairplot showed that the calculated 
differences between the values can be used to predict the outcome.
We will use those to fit and train the models.
The following section imports the libraries needed to fit and test different prediction models and compares 
the different models.
'''

# Models:
# Importing libraries.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.metrics import r2_score


# Linear Regression
from sklearn.linear_model import LinearRegression

# Using the numeric outcome for linear regression. In outcome: A - score means team 2 wins, + score means team 1 wins, and 0 menas draw.
# x and y are splitten into training and test data to test the models accuracy.
# As we could see from the pairplot, we will only use data that seems to be relevant to the outcome for x.
x = matches[['valueDif', 'GSDif', 'GCDif', 'rankingDif', 'winDif', 'pointsDif']].values
y = matches[['outcome']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)


# Fit the model
lm = LinearRegression()
lm.fit(x_train, y_train)

# Predict the outcome of the test data.
lm_predictions = lm.predict(x_test)
score = r2_score(y_test, lm_predictions)
print(score)

# Store the actual and predicted values to compare.
lm_results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': lm_predictions.flatten()})

# MAE, MSE, RMSE to show the accuracy of the predicted data.
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, lm_predictions))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, lm_predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, lm_predictions)))

# Reg plot to visualize the data.
lm_regplot = sns.regplot(x = matches.valueDif, y = matches.outcome)
plt.xlabel('Difference in squad values: +: team 1, -: team2')
plt.ylabel('Outcome: + for team 1, - for team 2, 0 for draw')
plt.title('Value Difference vs. Outcome')


# Variables to fit the model. Using the winner classification for the following models.
# x will be the classification: 1 for team1 win, 0 for draw, and -1 for team2 win. 
x = matches[['valueDif', 'GSDif', 'GCDif', 'rankingDif', 'winDif', 'pointsDif']].values
y = matches[['winner']].values


# KNN Classifier -- 64.4% accuracy.
from sklearn.neighbors import KNeighborsClassifier

# splitting the test and training data for the model. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# Initializing and fitting the model
knn_model = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn_model.fit(x_train, y_train)
# predict the outcome and probability of outcome.
knn_predictions = knn_model.predict(x_test)
knn_proba = knn_model.predict_proba(x_test)

knn_results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': knn_predictions.flatten()})

accuracy = accuracy_score(y_test, knn_predictions)
print(accuracy)
print(balanced_accuracy_score(y_test, knn_predictions))
print(knn_proba)


# GaussianNB Classifier -- 59.6% accuracy.
from sklearn.naive_bayes import GaussianNB

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state=5)

gau_clf = GaussianNB()
gau_clf.fit(x_train, y_train)
gau_predictions = gau_clf.predict(x_test)

gau_results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': gau_predictions.flatten()})

accuracy = accuracy_score(y_test, gau_predictions)
print(accuracy)
print(balanced_accuracy_score(y_test, gau_predictions))


# Decision Tree -- 40% accuracy
from sklearn.tree import DecisionTreeClassifier
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

tree = DecisionTreeClassifier()
tree_model = tree.fit(x_train, y_train)
tree_predictions = tree_model.predict(x_test)

tree_accuracy = accuracy_score(y_test, tree_predictions)
print(tree_accuracy)

# Decision Tree Visualization
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# Tree labels
predictors = ['valueDif', 'GSDif', 'GCDif', 'rankingDif', 'winDif', 'pointsDif']
classes = ['-1', '0', '1']

# Generate graphics
dot_data = StringIO()
export_graphviz(tree_model, out_file=dot_data, filled=True,
                rounded=True, special_characters=True,
                feature_names = predictors, class_names = classes)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Write file to file.
graph.write_png('WorldCup_DecisionTree.png')
Image(graph.create_png())


# Linear SVM --  57.3% accuracy
from sklearn import svm
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)

svm = svm.SVC(decision_function_shape='ovo').fit(x_train, y_train)
svm_prediction = svm.predict(x_test)

svm_results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': svm_prediction.flatten()})

svm_accuracy = accuracy_score(y_test, svm_prediction)
print(svm_accuracy)

'''
Comparing the accuracies of all models, we will use the KNN Classifier and Linear Regression to predict the 
World Cup outcomes.
The following section defines the match and match2 method. 'match' uses the KNN Classifier while 'match2' uses 
Linear Regression
'''

# Simulating a match using the KNN model
def match(teams, team1, team2, model):
    
    # Creating the data frame for the simulation teams' data.
    match = pd.DataFrame(columns=['team1TeanValue', 'team2TeamValue', 'team1GoalsScored', 'team2GoalsScored',
                                  'team1GoalsConc', 'team2GoalsConc', 'team1Win', 'team2Win',
                                  'team1Draw', 'team2Draw', 'team1Loss', 'team2Loss',
                                  'team1Ranking', 'team2Ranking', 'team1Points', 'team2Points'], index=[0])
    
    team1TeamValue = teams[teams.team == team1]['teamValue'].iloc[0]
    team2TeamValue = teams[teams.team == team2]['teamValue'].iloc[0]
    team1GoalsScored = teams[teams.team == team1]['gs'].iloc[0]
    team2GoalsScored = teams[teams.team == team2]['gs'].iloc[0]
    team1GoalsConc = teams[teams.team == team1]['gc'].iloc[0]
    team2GoalsConc = teams[teams.team == team2]['gs'].iloc[0]
    team1Win = teams[teams.team == team1]['win'].iloc[0]
    team2Win = teams[teams.team == team2]['win'].iloc[0]
    team1Draw = teams[teams.team == team1]['draw'].iloc[0]
    team2Draw = teams[teams.team == team2]['draw'].iloc[0]
    team1Loss = teams[teams.team == team1]['loss'].iloc[0]
    team2Loss = teams[teams.team == team2]['loss'].iloc[0]
    team1Ranking = teams[teams.team == team1]['ranking'].iloc[0]
    team2Ranking = teams[teams.team == team2]['ranking'].iloc[0]
    team1Points = teams[teams.team == team1]['points'].iloc[0]
    team2Points = teams[teams.team == team2]['points'].iloc[0]

    # Adding both teams' data to the data frame.
    match['team1Value'] = team1TeamValue 
    match['team2Value'] = team2TeamValue
    match['team1GoalsScored'] = team1GoalsScored
    match['team2GoalsScored'] = team2GoalsScored 
    match['team1GoalsConc'] = team1GoalsConc
    match['team2GoalsConc'] = team2GoalsConc 
    match['team1Win'] = team1Win
    match['team2Win'] = team2Win 
    match['team1Draw'] = team1Draw 
    match['team2Draw'] = team2Draw 
    match['team1Loss'] = team1Loss 
    match['team2Loss'] = team2Loss 
    match['team1Ranking'] = team1Ranking 
    match['team2Ranking'] = team2Ranking 
    match['team1Points'] = team1Points
    match['team2Points'] = team2Points
    
    # Providing the value differences that are used for the prediction.
    match['valueDif'] = match['team1Value'] - match['team2Value']
    match['GSDif'] = match['team1GoalsScored'] - match['team2GoalsScored']
    match['GCDif'] = match['team1GoalsConc'] - match['team2GoalsConc']
    match['winDif'] = match['team1Win'] - match['team2Win']
    match['pointsDif'] = match['team1Points'] - match['team2Points']
    match['rankingDif'] = match['team1Ranking'] - match['team2Ranking']
    
    # x values to predict the outcome.
    x_match = match[['valueDif', 'GSDif', 'GCDif', 'rankingDif', 'winDif', 'pointsDif']].values
    
    # Predict the outcome with the KNN model and print the probabilities for team1, team2, or draw. 
    prediction = model.predict(x_match)
    probability = model.predict_proba(x_match)
    print('Team', team1, 'probability', round((probability[0][2] * 100), 2), '%')
    print('Team', team2, 'probability', round((probability[0][0] * 100), 2), '%')
    print('Draw probability', round((probability[0][1] * 100), 2), '%')
    winner = None
    
    # Determening the winner and returning the winner.
    if prediction == '1':
        winner = team1
    elif prediction == '0':
        winner = 'Draw'
    elif prediction == '-1':
        winner = team2
        
    return winner


# Simulating match using Linear Regression. The outcome (goal difference) will be predicted and based on that
# the winner will be determened. 
def match2(teams, team1, team2, model):
    
    match = pd.DataFrame(columns=['team1TeanValue', 'team2TeamValue', 'team1GoalsScored', 'team2GoalsScored',
                                  'team1GoalsConc', 'team2GoalsConc', 'team1Win', 'team2Win',
                                  'team1Draw', 'team2Draw', 'team1Loss', 'team2Loss',
                                  'team1Ranking', 'team2Ranking', 'team1Points', 'team2Points'], index=[0])
    
    team1TeamValue = teams[teams.team == team1]['teamValue'].iloc[0]
    team2TeamValue = teams[teams.team == team2]['teamValue'].iloc[0]
    team1GoalsScored = teams[teams.team == team1]['gs'].iloc[0]
    team2GoalsScored = teams[teams.team == team2]['gs'].iloc[0]
    team1GoalsConc = teams[teams.team == team1]['gc'].iloc[0]
    team2GoalsConc = teams[teams.team == team2]['gs'].iloc[0]
    team1Win = teams[teams.team == team1]['win'].iloc[0]
    team2Win = teams[teams.team == team2]['win'].iloc[0]
    team1Draw = teams[teams.team == team1]['draw'].iloc[0]
    team2Draw = teams[teams.team == team2]['draw'].iloc[0]
    team1Loss = teams[teams.team == team1]['loss'].iloc[0]
    team2Loss = teams[teams.team == team2]['loss'].iloc[0]
    team1Ranking = teams[teams.team == team1]['ranking'].iloc[0]
    team2Ranking = teams[teams.team == team2]['ranking'].iloc[0]
    team1Points = teams[teams.team == team1]['points'].iloc[0]
    team2Points = teams[teams.team == team2]['points'].iloc[0]
    
    match['team1Value'] = team1TeamValue 
    match['team2Value'] = team2TeamValue 
    match['team1GoalsScored'] = team1GoalsScored 
    match['team2GoalsScored'] = team2GoalsScored 
    match['team1GoalsConc'] = team1GoalsConc 
    match['team2GoalsConc'] = team2GoalsConc 
    match['team1Win'] = team1Win
    match['team2Win'] = team2Win 
    match['team1Draw'] = team1Draw 
    match['team2Draw'] = team2Draw 
    match['team1Loss'] = team1Loss 
    match['team2Loss'] = team2Loss 
    match['team1Ranking'] = team1Ranking 
    match['team2Ranking'] = team2Ranking 
    match['team1Points'] = team1Points
    match['team2Points'] = team2Points
    
    match['valueDif'] = match['team1Value'] - match['team2Value']
    match['GSDif'] = match['team1GoalsScored'] - match['team2GoalsScored']
    match['GCDif'] = match['team1GoalsConc'] - match['team2GoalsConc']
    match['winDif'] = match['team1Win'] - match['team2Win']
    match['pointsDif'] = match['team1Points'] - match['team2Points']
    match['rankingDif'] = match['team1Ranking'] - match['team2Ranking']
    
    x_match = match[['valueDif', 'GSDif', 'GCDif', 'rankingDif', 'winDif', 'pointsDif']].values
    
    prediction = model.predict(x_match)
    
    winner = None
    
    # A negative value means team2 scores more and is therefore the winner. (And vice versa)
    if prediction > 0.1:
        winner = team1
    elif prediction < -0.1:
        winner = team2
    else:
        winner = 'Draw'
        
    return winner


'''
The following sections predict the outcome of the World Cup stage with the match and match2 method.
Running each line will display the winner and probability of each outcome.
Comments behind the lines display the winner and the 'points' section shows the result for each group.
'''

# Predicing the group stage with knn_model:

# Group A
print(match(teams, 'Brazil', 'Croatia', knn_model)) # Brazil
print(match(teams, 'Mexico', 'Cameroon', knn_model)) # Mexico
print(match(teams, 'Brazil', 'Mexico', knn_model)) # Brazil
print(match(teams, 'Cameroon', 'Croatia', knn_model)) # Croatia
print(match(teams, 'Cameroon', 'Brazil', knn_model)) # Draw
print(match(teams, 'Croatia', 'Mexico', knn_model)) # Croatia

# Points Group A: 
# Brazil: 7
# Croatia: 6
# Mexico: 3
# Cameroon: 1


# Group B:
print(match(teams, 'Spain', 'Netherlands', knn_model)) # Spain
print(match(teams, 'Chile', 'Australia', knn_model)) # Chile
print(match(teams, 'Australia', 'Netherlands', knn_model)) # Netherlands
print(match(teams, 'Spain', 'Chile', knn_model)) #  Spain
print(match(teams, 'Australia', 'Spain', knn_model)) # Spain
print(match(teams, 'Netherlands', 'Chile', knn_model)) # Netherlands

# Points:
# Spain: 9
# Netherlands: 6
# Chile: 3
# Australia: 0


# Group C:
print(match(teams, 'Colombia', 'Greece', knn_model)) # Colombia
print(match(teams, 'Ivory Coast', 'Japan', knn_model)) # Ivory Coast
print(match(teams, 'Colombia', 'Ivory Coast', knn_model)) # Colombia
print(match(teams, 'Japan', 'Greece', knn_model)) # Japan
print(match(teams, 'Japan', 'Colombia', knn_model)) # Colombia
print(match(teams, 'Greece', 'Ivory Coast', knn_model)) # Greece

# Points:
# Colombia: 9
# Greece: 3 
# Japan: 3
# Ivory Coast: 3


# Group D:
print(match(teams, 'Uruguay', 'Costa Rica', knn_model)) # Uruguay
print(match(teams, 'England', 'Italy', knn_model)) # England
print(match(teams, 'Uruguay', 'England', knn_model)) # England
print(match(teams, 'Italy', 'Costa Rica', knn_model)) # Italy
print(match(teams, 'Costa Rica', 'England', knn_model)) # Draw
print(match(teams, 'Italy', 'Uruguay', knn_model)) # Italy

# Points:
# England: 7
# Italy: 6
# Uruguay: 3
# Costa Rica: 1


# Group E:
print(match(teams, 'Switzerland', 'Ecuador', knn_model)) # Switzerland
print(match(teams, 'France', 'Honduras', knn_model)) # France
print(match(teams, 'Switzerland', 'France', knn_model)) # France
print(match(teams, 'Honduras', 'Ecuador', knn_model)) # Honduras
print(match(teams, 'Honduras', 'Switzerland', knn_model)) # Switzerland
print(match(teams, 'Ecuador', 'France', knn_model)) # Draw

# Points:
# France: 7
# Switzerland: 6
# Honduras: 3
# Ecuador: 1


# Group F:
print(match(teams, 'Argentina', 'Bosnia and Herzegovina', knn_model)) # Argentina
print(match(teams, 'Iran', 'Nigeria', knn_model)) # Draw
print(match(teams, 'Argentina', 'Iran', knn_model)) # Argentina
print(match(teams, 'Nigeria', 'Bosnia and Herzegovina', knn_model)) # Nigeria
print(match(teams, 'Nigeria', 'Argentina', knn_model)) # Draw
print(match(teams, 'Bosnia and Herzegovina', 'Iran', knn_model)) # Bosnia

# Points:
# Argentina: 7
# Nigeria: 5
# Bosnia: 3
# Iran: 1


# Group G:
print(match(teams, 'Germany', 'Portugal', knn_model)) # Germany
print(match(teams, 'Ghana', 'United States', knn_model)) # Ghana
print(match(teams, 'Germany', 'Ghana', knn_model)) # Germany
print(match(teams, 'United States', 'Portugal', knn_model)) # Draw
print(match(teams, 'Portugal', 'Ghana', knn_model)) # Portugal
print(match(teams, 'United States', 'Germany', knn_model)) # Germany

# Points: 
# Germany: 9
# Portugal: 4
# Ghana: 3
# Unites States: 1


# Group H: 
print(match(teams, 'Belgium', 'Algeria', knn_model)) # Belgium
print(match(teams, 'Russia', 'South Korea', knn_model)) # Russia
print(match(teams, 'Belgium', 'Russia', knn_model)) # Belgium
print(match(teams, 'South Korea', 'Algeria', knn_model)) # South Korea
print(match(teams, 'Algeria', 'Russia', knn_model)) # Russia
print(match(teams, 'South Korea', 'Belgium', knn_model)) # Draw

# Points:
# Belgium: 6
# Russia: 6
# South Korea: 4
# Algeria: 0



# comparing algorithms:
# Simulating games with Linear Regression.

# Group A:
print(match2(teams, 'Brazil', 'Croatia', lm)) # Brazil
print(match2(teams, 'Mexico', 'Cameroon', lm)) # Mexico
print(match2(teams, 'Brazil', 'Mexico', lm)) # Brazil
print(match2(teams, 'Cameroon', 'Croatia', lm)) # Croatia
print(match2(teams, 'Cameroon', 'Brazil', lm)) # Brazil
print(match2(teams, 'Croatia', 'Mexico', lm)) # Croatia

# Points Group A: 
# Brazil: 9
# Croatia: 6
# Mexico: 3
# Cameroon: 0

# Group B:
print(match2(teams, 'Spain', 'Netherlands', lm)) # Draw
print(match2(teams, 'Chile', 'Australia', lm)) # Chile
print(match2(teams, 'Australia', 'Netherlands', lm)) # Netherlands
print(match2(teams, 'Spain', 'Chile', lm)) #  Spain
print(match2(teams, 'Australia', 'Spain', lm)) # Spain
print(match2(teams, 'Netherlands', 'Chile', lm)) # Netherlands

# Points:
# Spain: 7
# Netherlands: 7
# Chile: 3
# Australia: 0


# Group C:
print(match2(teams, 'Colombia', 'Greece', lm)) # Colombia
print(match2(teams, 'Ivory Coast', 'Japan', lm)) # Japan
print(match2(teams, 'Colombia', 'Ivory Coast', lm)) # Colombia
print(match2(teams, 'Japan', 'Greece', lm)) # Draw
print(match2(teams, 'Japan', 'Colombia', lm)) # Colombia
print(match2(teams, 'Greece', 'Ivory Coast', lm)) # Greece

# Colombia: 9
# Greece: 4
# Japan: 4
# Ivory Coast: 3


# Group D:
print(match2(teams, 'Uruguay', 'Costa Rica', lm)) # Uruguay
print(match2(teams, 'England', 'Italy', lm)) # England
print(match2(teams, 'Uruguay', 'England', lm)) # England
print(match2(teams, 'Italy', 'Costa Rica', lm)) # Italy
print(match2(teams, 'Costa Rica', 'England', lm)) # England
print(match2(teams, 'Italy', 'Uruguay', lm)) # Daw

# Points:
# England: 9
# Italy: 4
# Uruguay: 4
# Costa Rica: 0


# Group E:
print(match2(teams, 'Switzerland', 'Ecuador', lm)) # Switzerland
print(match2(teams, 'France', 'Honduras', lm)) # France
print(match2(teams, 'Switzerland', 'France', lm)) # France
print(match2(teams, 'Honduras', 'Ecuador', lm)) # Ecuador
print(match2(teams, 'Honduras', 'Switzerland', lm)) # Switzerland
print(match2(teams, 'Ecuador', 'France', lm)) # France

# Points:
# France: 9
# Switzerland: 6
# Honduras: 0
# Ecuador: 3


# Group F:
print(match2(teams, 'Argentina', 'Bosnia and Herzegovina', lm)) # Argentina
print(match2(teams, 'Iran', 'Nigeria', lm)) # Iran
print(match2(teams, 'Argentina', 'Iran', lm)) # Argentina
print(match2(teams, 'Nigeria', 'Bosnia and Herzegovina', lm)) # Bosnia
print(match2(teams, 'Nigeria', 'Argentina', lm)) # Argentina
print(match2(teams, 'Bosnia and Herzegovina', 'Iran', lm)) # Bosnia

# Points:
# Argentina: 9
# Nigeria: 0
# Bosnia: 6
# Iran: 3


# Group G:
print(match2(teams, 'Germany', 'Portugal', lm)) # Germany
print(match2(teams, 'Ghana', 'United States', lm)) # United States
print(match2(teams, 'Germany', 'Ghana', lm)) # Germany
print(match2(teams, 'United States', 'Portugal', lm)) # Portugal
print(match2(teams, 'Portugal', 'Ghana', lm)) # Portugal
print(match2(teams, 'United States', 'Germany', lm)) # Germany    
   
# Points: 
# Germany: 9
# Portugal: 6
# Ghana: 0
# Unites States: 3
 
    
# Group H:
print(match2(teams, 'Belgium', 'Algeria', lm)) # Belgium
print(match2(teams, 'Russia', 'South Korea', lm)) # Russia
print(match2(teams, 'Belgium', 'Russia', lm)) # Draw
print(match2(teams, 'South Korea', 'Algeria', lm)) # South Korea
print(match2(teams, 'Algeria', 'Russia', lm)) # Russia
print(match2(teams, 'South Korea', 'Belgium', lm)) # Belgium
    
# Points:
# Belgium: 7
# Russia: 7
# South Korea: 3
# Algeria: 0


'''
Since there are no draws during the knock out stage of the World Cup, the draw outcome is deleted and the KNN
model is trained accondingly.
To be able to show the probabilities and due to the high accuracy, we will only use KNN for the knock out stage. 
For simplicity the data is prepared again in this section so you can only run this section if you only want to 
see the knock out stages. However, the libraries at the top of this file need to be imported.
'''

# Knock out stages:
# No more draws

# All steps to prepare the data are the same as before only draw results are stripped and the model is 
# trained without draws.
teams = pd.read_csv('/home/michael/Documents/SoccerAnalysisProject/Data/teamStats.csv')

matches = pd.read_csv('/home/michael/Documents/SoccerAnalysisProject/Data/allresults.csv')

matches = matches.drop(['city', 'tournament', 'country', 'neutral'], axis = 1)
matches.index = pd.DatetimeIndex(matches.date).year
matches = matches.drop('date', 1)

matches = matches.loc[2013:2014]
dfTeams = list(teams.team.unique())
matches = matches.reset_index()

for index, row in matches.iterrows():
    if row.home_team not in dfTeams:
        matches.loc[index, 'home_team'] = None
    if row.away_team not in dfTeams:
        matches.loc[index, 'away_team'] = None
        
matches = matches.dropna()

matches['homeTeamValue'] = matches.home_team.map(teams.set_index('team')['teamValue'].to_dict())
matches['awayTeamValue'] = matches.away_team.map(teams.set_index('team')['teamValue'].to_dict())
matches['homeTeamGoalsScored'] = matches.home_team.map(teams.set_index('team')['gs'].to_dict())
matches['awayTeamGoalsScored'] = matches.away_team.map(teams.set_index('team')['gs'].to_dict())
matches['homeTeamGoalsConc'] = matches.home_team.map(teams.set_index('team')['gc'].to_dict())
matches['awayTeamGoalsConc'] = matches.away_team.map(teams.set_index('team')['gc'].to_dict())
matches['homeTeamRanking'] = matches.home_team.map(teams.set_index('team')['ranking'].to_dict())
matches['awayTeamRanking'] = matches.away_team.map(teams.set_index('team')['ranking'].to_dict())
matches['homeWins'] = matches.home_team.map(teams.set_index('team')['win'].to_dict())
matches['awayWins'] = matches.away_team.map(teams.set_index('team')['win'].to_dict())
matches['homeDraws'] = matches.home_team.map(teams.set_index('team')['draw'].to_dict())
matches['awayDraws'] = matches.away_team.map(teams.set_index('team')['draw'].to_dict())
matches['homeLoss'] = matches.home_team.map(teams.set_index('team')['loss'].to_dict())
matches['awayLoss'] = matches.away_team.map(teams.set_index('team')['loss'].to_dict())
matches['homePoints'] = matches.home_team.map(teams.set_index('team')['points'].to_dict())
matches['awayPoints'] = matches.away_team.map(teams.set_index('team')['points'].to_dict())

matches['outcome'] = matches.home_score - matches.away_score
matches['winner'] = None
matches['winner'][matches.outcome > 0] = '1'
matches['winner'][matches.outcome < 0] = '-1'
matches['winner'][matches.outcome == 0] = None
matches['winner'] = pd.Categorical(matches['winner'])

# Drop Draws
matches = matches.dropna()
# Reset index
matches = matches.reset_index()
matches = matches.drop(['index'], axis=1)

matches['valueDif'] = matches.homeTeamValue - matches.awayTeamValue
matches['GSDif'] = matches.homeTeamGoalsScored - matches.awayTeamGoalsScored
matches['GCDif'] = matches.homeTeamGoalsConc - matches.awayTeamGoalsConc
matches['rankingDif'] = matches.homeTeamRanking - matches.awayTeamRanking
matches['winDif'] = matches.homeWins - matches.awayWins
matches['drawDif'] = matches.homeDraws - matches.awayDraws
matches['lossDif'] = matches.homeLoss - matches.awayLoss
matches['pointsDif'] = matches.homePoints - matches.awayPoints
    
# Training KNN without draws
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

x = matches[['valueDif', 'GSDif', 'GCDif', 'rankingDif', 'winDif', 'pointsDif']].values
y = matches[['winner']].values


# Dropping the draw outcome imporves the accuracy. 
from sklearn.neighbors import KNeighborsClassifier
 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5)

knn_model = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn_model.fit(x_train, y_train)

knn_predictions = knn_model.predict(x_test)
knn_proba = knn_model.predict_proba(x_test)

accuracy = accuracy_score(y_test, knn_predictions)
print(accuracy)
print(balanced_accuracy_score(y_test, knn_predictions))
print(knn_proba)

# Resuming k.o. stage solely with KNN.
# The ko_match method is basically the same match method only without the draw outcome.
def ko_match(teams, team1, team2, model):
    
    match = pd.DataFrame(columns=['team1TeanValue', 'team2TeamValue', 'team1GoalsScored', 'team2GoalsScored',
                                  'team1GoalsConc', 'team2GoalsConc', 'team1Win', 'team2Win',
                                  'team1Draw', 'team2Draw', 'team1Loss', 'team2Loss',
                                  'team1Ranking', 'team2Ranking', 'team1Points', 'team2Points'], index=[0])
    
    team1TeamValue = teams[teams.team == team1]['teamValue'].iloc[0]
    team2TeamValue = teams[teams.team == team2]['teamValue'].iloc[0]
    team1GoalsScored = teams[teams.team == team1]['gs'].iloc[0]
    team2GoalsScored = teams[teams.team == team2]['gs'].iloc[0]
    team1GoalsConc = teams[teams.team == team1]['gc'].iloc[0]
    team2GoalsConc = teams[teams.team == team2]['gs'].iloc[0]
    team1Win = teams[teams.team == team1]['win'].iloc[0]
    team2Win = teams[teams.team == team2]['win'].iloc[0]
    team1Draw = teams[teams.team == team1]['draw'].iloc[0]
    team2Draw = teams[teams.team == team2]['draw'].iloc[0]
    team1Loss = teams[teams.team == team1]['loss'].iloc[0]
    team2Loss = teams[teams.team == team2]['loss'].iloc[0]
    team1Ranking = teams[teams.team == team1]['ranking'].iloc[0]
    team2Ranking = teams[teams.team == team2]['ranking'].iloc[0]
    team1Points = teams[teams.team == team1]['points'].iloc[0]
    team2Points = teams[teams.team == team2]['points'].iloc[0]
    
    match['team1Value'] = team1TeamValue 
    match['team2Value'] = team2TeamValue
    match['team1GoalsScored'] = team1GoalsScored
    match['team2GoalsScored'] = team2GoalsScored 
    match['team1GoalsConc'] = team1GoalsConc
    match['team2GoalsConc'] = team2GoalsConc 
    match['team1Win'] = team1Win
    match['team2Win'] = team2Win 
    match['team1Draw'] = team1Draw 
    match['team2Draw'] = team2Draw 
    match['team1Loss'] = team1Loss 
    match['team2Loss'] = team2Loss 
    match['team1Ranking'] = team1Ranking 
    match['team2Ranking'] = team2Ranking 
    match['team1Points'] = team1Points
    match['team2Points'] = team2Points
    
    match['valueDif'] = match['team1Value'] - match['team2Value']
    match['GSDif'] = match['team1GoalsScored'] - match['team2GoalsScored']
    match['GCDif'] = match['team1GoalsConc'] - match['team2GoalsConc']
    match['winDif'] = match['team1Win'] - match['team2Win']
    match['pointsDif'] = match['team1Points'] - match['team2Points']
    match['rankingDif'] = match['team1Ranking'] - match['team2Ranking']
    
    x_match = match[['valueDif', 'GSDif', 'GCDif', 'rankingDif', 'winDif', 'pointsDif']].values
    
    prediction = model.predict(x_match)
    probability = model.predict_proba(x_match)
    print('Team', team1, 'probability', round((probability[0][1] * 100), 2), '%')
    print('Team', team2, 'probability', round((probability[0][0] * 100), 2), '%')
    winner = None
    
    if prediction == '1':
        winner = team1
    elif prediction == '-1':
        winner = team2
        
    return winner

'''
This section runs all matches during the knock out stage of the tournament. Running each line will show the winner
and the probability of each outcome again.
'''
# 1/8 Finals
# 1A vs. 2B:
print(ko_match(teams, 'Brazil', 'Netherlands', knn_model)) # Brazil

# 1C vs 2D
print(ko_match(teams, 'Colombia', 'Italy', knn_model)) # Italy

# 1E vs 2F
print(ko_match(teams, 'France', 'Nigeria', knn_model)) # France

# 1G vs 2H 
print(ko_match(teams, 'Germany', 'Russia', knn_model)) # Germany

# 1B vs 2A
print(ko_match(teams, 'Spain', 'Croatia', knn_model)) # Spain

# 1D vs 2C
print(ko_match(teams, 'England', 'Greece', knn_model)) # England

# 1F vs 2E
print(ko_match(teams, 'Argentina', 'Switzerland', knn_model)) # Argentina

# 1H vs 2G
print(ko_match(teams, 'Belgium', 'Portugal', knn_model)) # Belgium


# 1/4 Finals:
print(ko_match(teams, 'Brazil', 'Italy', knn_model)) # Brazil
print(ko_match(teams, 'France', 'Germany', knn_model)) # Germany
print(ko_match(teams, 'Spain', 'England', knn_model)) # Spain
print(ko_match(teams, 'Argentina', 'Belgium', knn_model)) # Argentina

# 1/2 Finals: 
print(ko_match(teams, 'Brazil', 'Germany', knn_model)) # Germany
print(ko_match(teams, 'Spain', 'Argentina', knn_model)) # Spain

# 3th place:
print(ko_match(teams, 'Brazil', 'Argentina', knn_model)) # Brazil

# Final:
print(ko_match(teams, 'Germany', 'Spain', knn_model)) # Spain

