
from .basic_function import *

def default_model_classifier(X,y,start,end,model_name):
        Listing=[]
        preprocessor= data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
        for i in range(start,end):
        
        # ****************Feature Seclortot**********************************************
                if model_name=="RandomForestClassifier":
                    model_selector = Pipeline(
                                steps=[("preprocessor", preprocessor),
                                ("feature", SelectKBest(f_regression,k=i)),
                                ("classifier", RandomForestClassifier())]
                            )
                    parms={
                "classifier__n_estimators":[100,150],
                
                # "classifier__criterion":["gini"],
                'classifier__min_samples_split':[1,2,3,4,5,7,8,9,10],
                'classifier__max_features':["sqrt", "log2", None],
                         
                    }
                elif model_name=="ExtraTreeClassifier":
                    parms={
                "classifier__splitter":["random", "best"],
                "classifier__max_features":["sqrt", "log2", "auto"],
                'classifier__max_depth':[10,15,20,25,30],
            } 
                    model_selector = Pipeline(
                            steps=[("preprocessor", preprocessor),
                            ("feature", SelectKBest(f_regression,k=i)),
                            ("classifier", ExtraTreeClassifier())]
                        )
                else:
                    print("Not working deafult model")
                    print("Not working deafult model")
                    
                feature_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=i))])
                feature_selector.fit(xtrain,ytrain)
        # ****************Model Seclortot**********************************************
            
                model_selector.fit(xtrain,ytrain)
        # *********************************Hyper Parametet***********************************
                grid=GridSearchCV(model_selector,parms,cv=4,n_jobs=-1,verbose=3)
                grid.fit(xtrain,ytrain)
                feature=grid.best_params_
                model=grid.best_estimator_
                # print(f"R2score of Result Modelwith best estimator is : {result_model}")
        #****************************Result Generation ******************************
                clas_report_parameter,conf_mat_parameter=result_evaluator_classfication(model,xtest,ytest)
                # print(classification_report(ypred,ytest))
                print("Confusion Matrix is With Best Perimator:" , conf_mat_parameter)
                # st.write(confusion_matrix(ypred,ytest))
                print("CLassification Report is With Best Perimator :",clas_report_parameter)
                clas_report,conf_mat_=result_evaluator_classfication(model_selector,xtest,ytest)
                print("Confusion Matrix is Without Best Perimator:" , conf_mat_parameter)
                # st.write(confusion_matrix(ypred,ytest))
                print("CLassification Report is  Without Best Perimator:",clas_report_parameter)
                
            
        #*********************************Working on features****************************
                xopt=feature_selector.get_feature_names_out()
                feature_selection=[]
                for x in xopt:
                    feature_selection.append(x.split("__")[1])
            
                # int("best feature")
                print("best feature")
                print("best feature")
                print(feature_selection)
                # print("**********new********************")

        # ********************Colecting Data--***********************************************
                Listing.append({
                    "i":i,
                    # "Error":result,
                    # "Error_model":result_model,
                    "columns":feature_selection,
                    "parameter":feature
                })
        print("model Name")
        print("model Name")
        print("model Name")
        print("model Name")
        print(model_name)
        return Listing
 


def default_model_regression(X,y,start,end,model_name):
        Listing=[]
        preprocessor= data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
    
        for i in range(start,end):
        # ****************Feature Seclortot**********************************************
                if model_name=="RandomForestRegressor":
                    model_selector = Pipeline(
                                steps=[("preprocessor", preprocessor),
                                ("feature", SelectKBest(f_regression,k=i)),
                                ("classifier", RandomForestRegressor())]
                            )
                    parms={
                "classifier__n_estimators":[100,150,200,250,270,300],
                "classifier__criterion":["squared_error", "absolute_error", "poisson"],
                'classifier__min_samples_split':[1,2,3,4,5,7,8,9,10],
                'classifier__max_features':["sqrt", "log2", None],
                         
                    }
                elif model_name=="ExtraTreeRegressor":
                    parms={
                "classifier__splitter":["random", "best"],
                "classifier__max_features":["sqrt", "log2", None],
                'classifier__max_depth':[10,15,20,25,30],
            } 
                    model_selector = Pipeline(
                            steps=[("preprocessor", preprocessor),
                            ("feature", SelectKBest(f_regression,k=i)),
                            ("classifier", ExtraTreeRegressor())]
                        )
                elif model_name=="LogisticRegression":
                    parms={
                    "classifier__penalty":["l1", "l2", "elasticnet", None],
                    "classifier__solve":["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                    'classifier__C':[1,2,3],
                } 
                    model_selector = Pipeline(
                            steps=[("preprocessor", preprocessor),
                            ("feature", SelectKBest(f_regression,k=i)),
                            ("classifier", ExtraTreeRegressor())]
                        )
                elif model_name=="KKNeighbors":
                    parms={
                    "classifier__weights":["uniform", "distance"],
                    "classifier__algorithm":["auto", "ball_tre", "kd_tree", "brute"],
                    'classifier__n_neighbors':[2,3,4,5,6,7],}
                     
                    model_selector = Pipeline(
                            steps=[("preprocessor", preprocessor),
                            ("feature", SelectKBest(f_regression,k=i)),
                            ("classifier", ExtraTreeRegressor())]
                        )
            
        
                else:
                   
                    print("Not working deafult model")
                    print("Not working deafult model")
                    print("Not working deafult model")
                     
                feature_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=i))])
                feature_selector.fit(xtrain,ytrain)
        # ****************Model Seclortot**********************************************
                model_selector.fit(xtrain,ytrain)
        # *********************************Hyper Parametet***********************************
                grid=GridSearchCV(model_selector,parms,cv=4,n_jobs=-1,verbose=3)
                grid.fit(xtrain,ytrain)
                feature=grid.best_params_
                model=grid.best_estimator_
                ypred_model=model.predict(xtest)
                result_model=r2_score(ytest,ypred_model)
                result_parameter,mae_parameter=result_evaluator_regressor(model,xtest,ytest)
                
                print(f"R2 score with best parameter{result_parameter}")
                print(f"Mean Absoulte Error with best parameter{mae_parameter}")
                
        #****************************Result Generation ******************************
                result,mae=result_evaluator_regressor(model_selector,xtest,ytest)
                

                print(f"Accuracy without  Best parameter{result}")
                print(f"Mean Absoulte Error without  best parameter{mae}")
            
        #*********************************Working on features****************************
                xopt=feature_selector.get_feature_names_out()
                feature_selection=[]
                for x in xopt:
                    feature_selection.append(x.split("__")[1])
            
                print("The feature Selection are as follow-:")
                print(feature_selection)
                print("Hypre Paramerter are as follow-:")
                print(feature)
                
        # ********************Colecting Data--***********************************************
                Listing.append({
                    "i":i,
                    "result":result,
                    "mae":mae,
                    "result_parameter":result_parameter,
                    "mae_parameter":mae_parameter,
                #     "Error_model":result_model,
                    "columns":feature_selection,
                    "parameter":feature
                })
        print(Listing)
        return Listing