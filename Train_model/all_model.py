from .basic_function import *
import random
import joblib
from .models import TrainedModel
import math
import datetime
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import NearMiss,RandomUnderSampler
def model_saving(model,remaining_url,name,model_name):
    base_url="media/"
    full_url=base_url+remaining_url
    id_generator=random.randint(1000,9999)
    print(f"The Generate did is {id_generator}")
    joblib.dump(model, open(full_url, 'wb'))
    utime=current_time = datetime.datetime.now().strftime("%H:%M:%S")
    udate =datetime.date.today()
    print("Udate")
    print("Udate")
    print("Udate")
    print("Udate")
    print("Udate")
    print("Udate")
    result=TrainedModel(username=name,location=full_url,modelname=model_name,uniqueid=id_generator,date=udate,time=utime)
    result.save()
    return id_generator

def all_regression(X,y,start,end,parms,model_name):
        # print("All Modle running")
        Listing=[]
        res2=-9999

        preprocessor= data_preprocessor(X,y)
        # if sampling=="SMOTE":
        #        X,y=SMOTE().fit_resample(X,y)
        # elif sampling=="ADASYN":
        #         X,y=ADASYN().fit_resample(X,y)
        # elif sampling=="RandomOverSampler":
        #         X,y=RandomOverSampler().fit_resample(X,y)
        # elif sampling=="RandomUnderSampler":
        #         X,y=RandomUnderSampler().fit_resample(X,y)
        # elif sampling=="NearMiss":
        #         X,y=NearMiss().fit_resample(X,y)


        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
  
        for i in range(start,end):
        # ****************Feature Seclortot**********************************************
                feature_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=i))])
                feature_selector.fit(xtrain,ytrain)
        # ****************Model Seclortot**********************************************
                model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=i)),
                    ("classifier", model_name)]
                )
                model_selector.fit(xtrain,ytrain)
        # *********************************Hyper Parametet***********************************
                grid=GridSearchCV(model_selector,parms,cv=4,n_jobs=-1,verbose=3)
                grid.fit(xtrain,ytrain)
                feature=grid.best_params_
                model=grid.best_estimator_

                result_parameter,mae_parameter=result_evaluator_regressor(model,xtest,ytest)
             
        #****************************Result Generation ******************************
                # ypred=model_selector.predict(xtest)
                # result=r2_score(ytest,ypred)
                result,mae=result_evaluator_regressor(model_selector,xtest,ytest)
           
            
        #*********************************Working on features****************************
                xopt=feature_selector.get_feature_names_out()
                feature_selection=[]
                for x in xopt:
                    feature_selection.append(x.split("__")[1])
            
                print("The feature Selection are as follow-:")
                print(feature_selection)
                print("Hypre Paramerter are as follow-:")
                print(feature)
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
                if math.floor(res2)<math.floor(result_parameter):
                        download_model=model
                        res2=result_parameter
                        # print(i)
                        # print(("after",math.floor(res2),math.floor(result_parameter)))
        # print(Listing)
        return Listing,download_model

def all_classification_model(X,y,start,end,parms,model_name,sampling="RandomOverSampler"):
        Listing=[]
        temp=1
        res2=-9999
        preprocessor=data_preprocessor(X,y)
        if sampling=="SMOTE":
                pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('oversampler', SMOTE())
                         ])
        elif sampling=="NearMiss":
                pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('oversampler', NearMiss())
                         ])
        elif sampling=="RandomOverSampler":
                 pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                 ('oversampler', RandomOverSampler())
                                ])
        elif sampling=="RandomUnderSampler":
                 pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                 ('oversampler', RandomUnderSampler())
                                ])
        
               
        
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
  
        for i in range(start,end):
        # ****************Feature Seclortot**********************************************
                feature_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=i))])
                feature_selector.fit(xtrain,ytrain)
        # ****************Model Selector**********************************************
                model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=i)),
                    ("classifier", model_name)]
                )
                model_selector.fit(xtrain,ytrain)
        # *********************************Hyper Parametet***********************************
                grid=GridSearchCV(model_selector,parms,cv=4,n_jobs=-1,verbose=3)
                grid.fit(xtrain,ytrain)
                feature=grid.best_params_
                model=grid.best_estimator_
                # ypred_model=model_selector.predict(xtest)
                
        #****************************Result Generation ******************************
                clas_report_parameter,accuracy_parameter=result_evaluator_classfication(model,xtest,ytest,temp)
                # temp=temp+1
                
                print("Confusion Matrix is With Best Perimator:" , accuracy_parameter)
                # st.write(confusion_matrix(ypred,ytest))
                print("CLassification Report is With Best Perimator :",clas_report_parameter)
                clas_report,accuracy=result_evaluator_classfication(model_selector,xtest,ytest,temp,hyper="no")
                temp=temp+1
                print("Confusion Matrix is Without Best Perimator:" , accuracy)
                # st.write(confusion_matrix(ypred,ytest))
                print("CLassification Report is  Without Best Perimator:",clas_report)
                # st.write(classification_report(ypred,ytest))
                
        #*********************************Working on features****************************
                xopt=feature_selector.get_feature_names_out()
                feature_selection=[]
                for x in xopt:
                    feature_selection.append(x.split("__")[1])
                print("The feature Selection are as follow-:")
                print(feature_selection)
                print("The Hyper Parameter are as follow -:")
                print(feature)
                print(feature_selection)
                # st.write("************************")
                # st.write(f"Iteration Number is{i} ")
                # print(f"***********************--{i}******************")
                # print("**********new********************")

        # ********************Colecting Data--***********************************************
                Listing.append({
                    "i":i,
                    "accuracypara":accuracy_parameter,
                    "clas_report_para":clas_report_parameter,
                    "accuracy":accuracy,
                    # "Error_model":result_model,
                    "columns":feature_selection,
                    "parameter":feature
                })
                print(";isting")
                print(";isting")
                print(";isting")
                print(Listing)
                if math.floor(res2)<math.floor(accuracy_parameter):
                        download_model=model
                        res2=accuracy_parameter
        return Listing ,download_model

def creating_hyper_paramerter(parms,column):
                print("inside hyper")
                if len(column)>1:
                    try:
                        parms["classifier__"+column[0]]=column[1:]
                        print("insid all model")
                        print("insid all model")
                        print(parms)
                        return parms
                    except Exception as e:
                        print(Exception)  

                            



