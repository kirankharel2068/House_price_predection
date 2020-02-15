# House_price_predection

This project is done to make housing predection implementing various Machine Learning libraries. 
I have tried to go through various phases of Data science life cycle to make the predections which are as follows:
1.  Data Import and Exploratory Data Analysis
    In this stage I have imported data from local directory (main source: Kaggle) and then performed EDA to have a better understanding
    of the data. 
2. Data Cleaning 
    In this stage I have analyzed various features of data such as handling null values, handling skewness of the data, removing outliers     etc. 
    
3. Feature Engineering and Data Preparation
    In this stage, I have handled various categorical features using Label Encoder and have also performed feature engineering inorder to     improve the performance of our Machine learning model.
4. Model fitting
    I have mainly focused on this section, focusing on implementation of various machine learning algorithms. Apart from this, I have also tried using Pipelines and Staking techniques along with Hypertuning       optimization using Gridsearch CV 
5. Making Predictions and Evalutiong performance
    In this stage all the models are used to make predectiona and their performance has been compared using visualization. The metrics         used  for evaluation is mean squared error
    
##list of files
-----------------
* EDA and Feature Engineering.ipynb: Performed Exploratory Data analysis and Feature engineering to prepared data for modeling.
* HousePrice_prediction_test.ipynb: This file contains all the stages as it was performed to make comparision and test the entire process.
* Model_Bagging.ipynb: Various bagging techniques including Random Forest
* Model_Ensambles_Voting.ipynb: Uses Voting Regressor to make predictions
* Model_Stacking.ipynb: Used Stacking ensembles to make predicitions.
* Modeling_Boosting.ipynb: Includes various types of boosting like AdaBoost, Gradient Boosting etc.
* Modeling_pipelines.ipynb: Used Pipelines to build a model using various ML models like SVM, Ridge, Decision Trees etc.
* model_ANN.ipynb: Implemented Artificial Neural Network to make predictions.
* utils.py: Constains all the common functions used for the project.
