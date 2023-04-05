
from .basic_function import *

def default_model_classifier(X,y,start,end,model_name_list):
        Listing=[]
        temp=1
        preprocessor= data_preprocessor(X,y)
        sampling_list=["SMOTE","RandomUnderSampler"]
        for sampling in sampling_list:
                X_resampled,y_resampled=dataSampling(sampling,X,y)
                xtrain,xtest,ytrain,ytest=train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=45)
                for model_name in model_name_list:
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
                            'classifier__min_samples_split':[1,2,],
                            'classifier__max_features':["sqrt"],
                                    
                                }
                            elif model_name=="LogisticRegression":
                                parms={
                                "classifier__penalty":["l1", "l2", "elasticnet", None],
                                "classifier__solver":["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                                'classifier__C':[1,2,3],
                                        } 
                                model_selector = Pipeline(
                                        steps=[("preprocessor", preprocessor),
                                        ("feature", SelectKBest(f_regression,k=i)),
                                        ("classifier", LogisticRegression())]
                                    )
                            elif model_name=="KNeighbors":
                                parms={
                                "classifier__weights":["uniform", ],
                                "classifier__algorithm":["auto",],
                                'classifier__n_neighbors':[6,7],}
                                
                                model_selector = Pipeline(
                                        steps=[("preprocessor", preprocessor),
                                        ("feature", SelectKBest(f_regression,k=i)),
                                        ("classifier", KNeighborsClassifier())]
                                    )
                            elif model_name=="DecisionTree":
                                    parms={
                                    "classifier__criterion":["gini", "entropy", "log_loss"  ],
                                    "classifier__splitter":["best", "random"],
                                        
                                    }
                                    model_selector = Pipeline(
                                            steps=[("preprocessor", preprocessor),
                                            ("feature", SelectKBest(f_regression,k=i)),
                                            ("classifier", DecisionTreeClassifier())]
                                        )
                            elif model_name=="GradientBoosting":
                                    parms={
                                    "classifier__loss":["log_loss", "deviance", "exponential"],
                                    "classifier__n_estimators":[90,95,100]
                                        
                                    }
                                    model_selector = Pipeline(
                                            steps=[("preprocessor", preprocessor),
                                            ("feature", SelectKBest(f_regression,k=i)),
                                            ("classifier", GradientBoostingClassifier())]
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
                            clas_report_parameter,accuracy_parameter=result_evaluator_classfication(model,xtest,ytest,temp)
                            # print(classification_report(ypred,ytest))
                            # print("Confusion Matrix is With Best Perimator:" , conf_mat_parameter)
                            # st.write(confusion_matrix(ypred,ytest))
                            print("CLassification Report is With Best Perimator :",clas_report_parameter)
                            clas_report,accuracy=result_evaluator_classfication(model_selector,xtest,ytest,temp,hyper="no")
                            temp=temp+1
                            # print("Confusion Matrix is Without Best Perimator:" , conf_mat_parameter)
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
                                "accuracypara":accuracy_parameter,
                                "clas_report_para":clas_report_parameter,
                                "accuracy":accuracy,
                                # "Error_model":result_model,
                                "columns":feature_selection,
                                "parameter":feature,
                                "sampler":sampling,
                                "model":model_name,
                                })
                            
                    print("model Name")
                    print("model Name")
                    print("model Name")
                    print("model Name")
                    print(model_name)
                    print(Listing)
        return Listing,model
    


def default_model_regression(X,y,start,end,model_name_list):
        Listing=[]
        temp=start
        preprocessor= data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
        for model_name in model_name_list:
            print("inside deFULT")
            print("inside deFULT")
            print("inside deFULT")
            print("inside deFULT")
            print(model_name)
            print(f"satart{start}")
            start=temp
            print(start,temp)
            print(end)
            for i in range(start,end):
                    print(f"indide model{model_name}")
            # ****************Feature Seclortot**********************************************
                    if model_name=="RandomForestRegressor":
                        model_selector = Pipeline(
                                    steps=[("preprocessor", preprocessor),
                                    ("feature", SelectKBest(f_regression,k=i)),
                                    ("classifier", RandomForestRegressor())]
                                )
                        parms={
                    "classifier__n_estimators":[100,150],
                    
                    'classifier__max_features':["sqrt"],
                            
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
                    
                    elif model_name=="KNeighbors":
                        parms={
                        "classifier__weights":["uniform", ],
                        "classifier__algorithm":["auto",],
                        'classifier__n_neighbors':[6,7],}
                        
                        model_selector = Pipeline(
                                steps=[("preprocessor", preprocessor),
                                ("feature", SelectKBest(f_regression,k=i)),
                                ("classifier", KNeighborsRegressor())]
                            )
                    elif model_name=="DecisionTree":
                        parms={
                        "classifier__criterion":["squared_error", "friedman_mse", "absolute_error", "poisson" ],
                        "classifier__splitter":["best", "random"],
                              
                         }
                        model_selector = Pipeline(
                                steps=[("preprocessor", preprocessor),
                                ("feature", SelectKBest(f_regression,k=i)),
                                ("classifier", DecisionTreeRegressor())]
                            )
                    elif model_name=="GradientBoosting":
                        parms={
                        "classifier__loss":["squared_error", "absolute_error", "huber", "quantile"],
                        "classifier__n_estimators":[90,95,100]
                              
                         }
                        model_selector = Pipeline(
                                steps=[("preprocessor", preprocessor),
                                ("feature", SelectKBest(f_regression,k=i)),
                                ("classifier", GradientBoostingRegressor())]
                            )
                    
                
            
                    else:
                        
                        print("Not working deafult model")
                        print("Not working deafult model")
                        print("Not working deafult model")
                        print(model_name)
                        
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
                        "parameter":feature,
                        "model":model_name,
                    })
            print(Listing)
        return Listing,model