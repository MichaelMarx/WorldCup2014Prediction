# WorldCup2014Prediction
## Problem Statement
The code file contains code to predict the outcomes of the matches played during the 2014 World Cup. For that, the code imports the data (found in the Data folder) into Pandas DataFrames and prepares the data for further analytics. The challange was to find meaningful data from which the prediction models can accurately predict a soccer game's outcome. 
The file imports and prepares the data, fits and trains the prediction models, and predicts the outcome of the matches with the most accurate model. 

## Findings
Throughout the process, I found that calculating the differences between the playing teams and letting the model predict the outcome based on those differences produces a higher accuracy. The pairplot provided in the Pictures folder states the relationship between the provided data and shows what data is suited to be used to fit the models. Other data visualizations can be found in this folder as well. 

## How To Run The Code
The code may be run in the provided order. As the knockout stage of the tournament does not allow any draws, the prediction model is trained again after the simulation of the group stage. If you are only interested in the knockout stage, you can only run the second half of the code as I have provided the import and preparation of the data again for the knockout stage. In this case, you will have to import the libraries all the way on top of the file. The rest can be run in the provided order. 
Running every line of the simulated matches will provide the predicted winner together with the predicted probabilities of each outcome.
Needed libraries inlcude: 
- Pandas
- Numpy
- Seaborn
- Matplotlib.pyplot
- Sklearn
- IPython
- Pydotplus

## Examples of Visualizations (found in the Pictures Folder)
### BoxPlot 
![](/Pictures/boxplot.png)

### ScatterPlot
![](/Pictures/ScatterPlot.png)
