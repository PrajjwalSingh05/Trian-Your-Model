
from .basic_function import *


def download_random_forest_regressor(X,y,paramlist):
        #****************************Unpacking Tuple **************************************************
        input_col,Icriterion,Imaxfeature,Isamples_split,In_estimator,Imaxdepth_value=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", RandomForestRegressor(criterion=Icriterion,max_features=Imaxfeature,min_samples_split=Isamples_split,
                    n_estimators=In_estimator, max_depth=Imaxdepth_value))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        ypred=model_selector.predict(xtest)
        result=r2_score(ytest,ypred)
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print(result)
        return model_selector
def download_logistic_regresion(X,y,paramlist):
        #****************************Unpacking Tuple **************************************************
        input_col,ipenalty,imulti_class,isolver,ic_value=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", LogisticRegression(penalty=ipenalty,multi_class=imulti_class,solver=isolver,
                    C=ic_value, ))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        ypred=model_selector.predict(xtest)
        result=r2_score(ytest,ypred)
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print(result)
        

        return model_selector
def download_extratree_regresion(X,y,paramlist):
        #****************************Unpacking Tuple **************************************************
        input_col,icriterion,imaxfeature,indepth,isamplesplit=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", ExtraTreeRegressor(criterion=icriterion,max_features=imaxfeature,min_samples_split=isamplesplit,
                    max_depth=indepth ))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        ypred=model_selector.predict(xtest)
        result=r2_score(ytest,ypred)
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print(result)
        

        return model_selector
def download_knn_regression(X,y,paramlist):
        #****************************Unpacking Tuple **************************************************
        input_col,iweight,ialgorithm,kvalue=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", KNeighborsRegressor(n_neighbors=kvalue,weights=iweight,algorithm=ialgorithm,
                     ))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        ypred=model_selector.predict(xtest)
        result=r2_score(ytest,ypred)
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print(result)
        

        return model_selector
def download_svr_regression(X,y,paramlist):
        #****************************Unpacking Tuple **************************************************
        input_col,ikernal,igamma,degreevalue=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", SVR(kernel=ikernal,gamma=igamma,degree=degreevalue ))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        ypred=model_selector.predict(xtest)
        result=r2_score(ytest,ypred)
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print(result)
        

        return model_selector