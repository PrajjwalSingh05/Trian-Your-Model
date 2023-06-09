
from .basic_function import *
from imblearn.pipeline import Pipeline 

def download_random_forest_regressor(X,y,paramlist):
        Listing=[]
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
        result_parameter,mae_parameter=result_evaluator_regressor(model_selector,xtest,ytest)
        #*********************************************Result Generation ***************************************
        
        Listing.append({
                      
                    "result_parameter":result_parameter,
                    "mae_parameter":mae_parameter,
                #     "Error_model":result_model,
                    
                })
        return Listing,model_selector
def download_random_forest_classification(X,y,paramlist,testsize=0.2,randomstate=45):
        Listing=[]
        temp=1
        #****************************Unpacking Tuple **************************************************
        input_col,Icriterion,Imaxfeature,Isamples_split,In_estimator,Imaxdepth_value,sampling=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        X_resampled,y_resampled=dataSampling(sampling,X,y)
        
        xtrain,xtest,ytrain,ytest=train_test_split(X_resampled,y_resampled,test_size=testsize,random_state=randomstate)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", RandomForestClassifier(criterion=Icriterion,max_features=Imaxfeature,min_samples_split=Isamples_split,
                    n_estimators=In_estimator, max_depth=Imaxdepth_value))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        
        clas_report_parameter,accuracy_parameter=result_evaluator_classfication(model_selector,xtest,ytest,temp,hyper="no")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        Listing.append({
                #     "i":i,
                    "accuracypara":accuracy_parameter,
                    "clas_report_para":clas_report_parameter,
                })
        return Listing,model_selector
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
        result_parameter,mae_parameter=result_evaluator_regressor(model_selector,xtest,ytest)
        #*********************************************Result Generation ***************************************
        Listing.append({
                      
                    "result_parameter":result_parameter,
                    "mae_parameter":mae_parameter,
                #     "Error_model":result_model,
                    
                })
        

        return model_selector
# def download_extratree_regresion(X,y,paramlist):
#         #****************************Unpacking Tuple **************************************************
#         input_col,icriterion,imaxfeature,indepth,isamplesplit=paramlist
#         #****************************Data Preprocessing******************* ******************************
#         preprocessor= data_preprocessor(X,y)
#         xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
#         #*********************************************Creating Model***************************************
#         model_selector = Pipeline(
#                     steps=[("preprocessor", preprocessor),
#                     ("feature", SelectKBest(f_regression,k=input_col)),
#                     ("classifier", ExtraTreeRegressor(criterion=icriterion,max_features=imaxfeature,min_samples_split=isamplesplit,
#                     max_depth=indepth ))]
#                 )
#         model_selector.fit(xtrain,ytrain)
#         #*********************************************Result Generation ***************************************
#         ypred=model_selector.predict(xtest)
#         result=r2_score(ytest,ypred)
#         print("donload Model")
#         print("donload Model")
#         print("donload Model")
#         print("donload Model")
#         print(result)
        
#         return model_selector
def download_decision_regression(X,y,paramlist):
        Listing=[]
        #****************************Unpacking Tuple **************************************************
        input_col,ispliter,ucriterion,imaxfeature,iminsamplevalue,imaxdepthvalue=paramlist


      
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", DecisionTreeRegressor(splitter=ispliter,criterion=ucriterion,max_features=imaxfeature,min_samples_split=iminsamplevalue,
                     ))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        result_parameter,mae_parameter=result_evaluator_regressor(model_selector,xtest,ytest)
        #*********************************************Result Generation ***************************************
        
        Listing.append({
                      
                    "result_parameter":result_parameter,
                    "mae_parameter":mae_parameter,
                #     "Error_model":result_model,
                    
                })
        return Listing,model_selector

def download_knn_regression(X,y,paramlist):
        Listing=[]
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
        result_parameter,mae_parameter=result_evaluator_regressor(model_selector,xtest,ytest)
        #*********************************************Result Generation ***************************************
        
        Listing.append({
                      
                    "result_parameter":result_parameter,
                    "mae_parameter":mae_parameter,
                #     "Error_model":result_model,
                    
                })
        return Listing,model_selector
def download_knn_classfier(X,y,paramlist,testsize=0.2,randomstate=45):
        Listing=[]
        temp=1
        #****************************Unpacking Tuple **************************************************
        input_col,iweight,ialgorithm,kvalue,sampling=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        X_resampled,y_resampled=dataSampling(sampling,X,y)
        
        xtrain,xtest,ytrain,ytest=train_test_split(X_resampled,y_resampled,test_size=testsize,random_state=randomstate)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", KNeighborsClassifier(n_neighbors=kvalue,weights=iweight,algorithm=ialgorithm,
                     ))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        clas_report_parameter,accuracy_parameter=result_evaluator_classfication(model_selector,xtest,ytest,temp,hyper="no")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        # print(result)
        Listing.append({
                #     "i":i,
                    "accuracypara":accuracy_parameter,
                    "clas_report_para":clas_report_parameter,
        })
        
        return Listing,model_selector

def download_decision_classfier(X,y,paramlist,testsize=0.3,randomstate=45):

        Listing=[]
        temp=1
        #****************************Unpacking Tuple **************************************************
        input_col,ispliter,ucriterion,imaxfeature,iminsamplevalue,imaxdepthvalue,sampling=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        print("********************")
        print(sampling)
        
        X_resampled,y_resampled=dataSampling(sampling,X,y)
        
        xtrain,xtest,ytrain,ytest=train_test_split(X_resampled,y_resampled,test_size=testsize,random_state=randomstate)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", DecisionTreeClassifier(splitter=ispliter,criterion=ucriterion,max_features=imaxfeature,min_samples_split=iminsamplevalue,
                     ))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        clas_report_parameter,accuracy_parameter=result_evaluator_classfication(model_selector,xtest,ytest,temp,hyper="no")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        # print(result)
        Listing.append({
                #     "i":i,
                    "accuracypara":accuracy_parameter,
                    "clas_report_para":clas_report_parameter,
        })
        
        return Listing,model_selector

def download_logistic_classifier(X,y,paramlist,testsize=0.2,randomstate=45):
        Listing=[]
        temp=1
        temp=1
        #****************************Unpacking Tuple **************************************************
        input_col,ipenalty,imulti_class,isolver,ic_value,sampling=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        X_resampled,y_resampled=dataSampling(sampling,X,y)
        
        xtrain,xtest,ytrain,ytest=train_test_split(X_resampled,y_resampled,test_size=testsize,random_state=randomstate)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", LogisticRegression(penalty=ipenalty,multi_class=imulti_class,solver=isolver,
                    C=ic_value, ))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        clas_report_parameter,accuracy_parameter=result_evaluator_classfication(model_selector,xtest,ytest,temp,hyper="no")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        # print(result)
        Listing.append({
                #     "i":i,
                    "accuracypara":accuracy_parameter,
                    "clas_report_para":clas_report_parameter,
        })
        

        return Listing,model_selector
def download_svc_classfier(X,y,paramlist,testsize=0.2,randomstate=45):
        temp=1
        Listing=[]
        #****************************Unpacking Tuple **************************************************
        input_col,ikernal,igamma,degreevalue,sampling=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        X_resampled,y_resampled=dataSampling(sampling,X,y)
        
        xtrain,xtest,ytrain,ytest=train_test_split(X_resampled,y_resampled,test_size=testsize,random_state=randomstate)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", SVC(kernel=ikernal,gamma=igamma,degree=degreevalue ))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        clas_report_parameter,accuracy_parameter=result_evaluator_classfication(model_selector,xtest,ytest,temp,hyper="no")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        print("donload Model")
        # print(result)
        Listing.append({
                #     "i":i,
                    "accuracypara":accuracy_parameter,
                    "clas_report_para":clas_report_parameter,
        })

        return Listing,model_selector
def download_svr_regression(X,y,paramlist):
        Listing=[]
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
        result_parameter,mae_parameter=result_evaluator_regressor(model_selector,xtest,ytest)
        #*********************************************Result Generation ***************************************
        
        Listing.append({
                      
                    "result_parameter":result_parameter,
                    "mae_parameter":mae_parameter,
                #     "Error_model":result_model,
                    
                })
        return Listing,model_selector
def download_gradientboosting_regression(X,y,paramlist):
        Listing=[]
        #****************************Unpacking Tuple **************************************************
        input_col,icritrion,imaxfeature,ilearningrate,inestimator,imaxdepth,isamplesplit=paramlist
       
        #****************************Data Preprocessing******************* ******************************
        preprocessor= data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", GradientBoostingRegressor(criterion=icritrion,max_features=imaxfeature,learning_rate=ilearningrate,n_estimators=inestimator,max_depth=imaxdepth,min_samples_split=isamplesplit ))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        result_parameter,mae_parameter=result_evaluator_regressor(model_selector,xtest,ytest)
        #*********************************************Result Generation ***************************************
        
        Listing.append({
                      
                    "result_parameter":result_parameter,
                    "mae_parameter":mae_parameter,
                #     "Error_model":result_model,
                    
                })
        return Listing,model_selector