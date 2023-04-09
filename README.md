# Train Your Model -Website
### A Website that is used to train your machine learning model,provide graphic user interface to User
### In this project, I start by preprocessing the data by cleaning, transforming, and normalizing it. Then, I add a feature selection step to identify the most relevant features for the machine learning model. Next, I choose a machine learning algorithm based on the nature of the problem and the available data. Finally, I use hyperparameter tuning techniques, such as grid search or randomized search, to optimize the model's performance by selecting the best combination of hyperparameters


###  How The Machine Learning Model Work
For Better Understanding How the Machine Learnibng model is Being Created .I am gona use a dataset Of blood Donation Campeign
This is dataset Of Donation Of blood Ny peope in March Moth 
![step1](https://user-images.githubusercontent.com/83647407/230299722-dea98073-88dd-467d-9b30-10c230940773.png)

## Step1-:Convert All the column of the data into Numbercial Format since A machine learning alogirthm is based on numerical formula 
#### After COnverting the datset into Numercial column it is visible something like this-:

![step2](https://user-images.githubusercontent.com/83647407/230238910-469ea71c-e4f2-4e51-a2b1-08e82594bb3e.png)

## Step 2-:Feature Selection-:In this step we Remove Unwanted Column from the Dataset 
#### In the following exmaple we can see column name-:Unamed:0 act as a index which would be irrevent in the Machine Learning Model So we rempve that column
![Step3](https://user-images.githubusercontent.com/83647407/230239691-6214999a-2256-46f3-83c3-65c8db0a19d3.png)

## Step3-Choose a Model From The List of various Model
#### This is a List of Various Option Of machine leraning Model
![step4](https://user-images.githubusercontent.com/83647407/230300193-53fd6dc9-211d-446a-b638-077535b46c20.png)

## Step4-:Evaluate The Model 
#### In Regression Model we check The mean Squarred Error Of the model Lower is the value Higger is the accuracy,We can also Check r2 Score Higher the value of r2Score Higher is accuracy

##### In Classification Problem We Check The confusion Matrix And Classification Report Of the Model



## Step 5-: Choosing Hyperparemeter
#### After Choosing The Model We Do Hyperparamertering Of the Choosen Model To increase the Accury Of the Model As Heighest as Possible
#### Hyperparameter tuning is the process of selecting optimal values for parameters that cannot be learned by the model during training. It involves selecting values for model architecture, learning rates, regularization, and other settings to optimize performance on a specific task.



![pipeline](https://user-images.githubusercontent.com/83647407/230295224-e7f3c5d7-ba2d-4881-9744-345bb5d950df.png)




### Features: 
#### 1.A user-friendly interface to train a model with adjustable parameters and display training progress.
#### 2. Anyone Can train and Export Thier Machine Learning Model Without any prior Knowledge of MAchine Learning
#### 3. Provide Some default Database Which can be extracted by the user by specifying the number of records you need
#### 4.An option to download the trained model with a click of a button.

## Files Information:


#### 1.all_model.py-:
###### This Python file contains custom functions for regression and classification, which can be used for machine learning tasks to build predictive models.

#### 2.basic_function.py-:
###### This Python FIle Contain some of the  basic function such as preprocessor,result Evaluator

#### 3 Download_model.py-:
###### This Python File Contain The Function That are Requred to Download Machine Learning  Modelue by a single click

#### 4 default_model.py-:
###### This Python File COnatain The Feaflut menu bar tab Function


#### 5.Models.py-:
###### This File Contain data regrading the database of the Project 
#### 6 Views.py-:
###### It contain the Business Logic of the Project Like How data is being Fetched,saved and shown to the end user
