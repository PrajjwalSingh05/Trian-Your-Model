from django.shortcuts import render,redirect,HttpResponse
from django.contrib.auth.models import User,auth
from django.contrib.auth  import authenticate,login,logout
from django.contrib import messages
import requests
import pandas as pd
from bs4 import BeautifulSoup
import pickle
import csv
from .models import *
from .all_model  import *
from .default_model  import *
from .download_model import *
import joblib
import os
import random
# ***************************************mahine laring*****************

import pandas as pd
def admin_home(request):
    return render(request,"admin_home.html")

def index(request):
    return render(request,"index.html")
def logout(request):
    auth.logout(request)
    return redirect('/')
def login(request):
    error=""
    if request.method=='POST':
        
        u=request.POST['uname']
        p=request.POST['pswd']
        
        user=auth.authenticate(username=u,password=p)
        try:
            if user.is_staff:
                print("Inside  ")
                auth.login(request,user)
                messages.success(request,"Login Successfull")
                return redirect('admin_home')
                error="no"
            elif user is not None:
                auth.login(request,user)
                messages.success(request,"Login Successfull")
                return redirect('user_home')
                error="not"
            else:
                error="yes"
                messages.error(request,"Some error occurred")
        except:
            messages.error(request,"Invalid Login Credentials")
            error="yes"
    d={'error':error}
    return render(request, 'login.html',d)
def user_home(request):
    return render(request,"user_home.html")

def default_regression(request):
    Listing=[]
    download_status=0
    error=""
    # request.session['trp'] = 40
    # Listing=pd.DataFrame()
    df = pd.DataFrame()
    if request.method == 'POST' :
        print("using post method")
        pred_col=request.POST["predcol"]
        model_type=request.POST["mtype"]
        end=int(request.POST["size1"])
        # end=int(end)
        # print(type(end))
        u=request.FILES['datafile']
        df=pd.read_csv(u)
        maxcol=df.shape[1]-2
        # print(df.head())
        try:
            X=df.drop(columns=pred_col)
            y=df[pred_col]
            if parameter_checkup(end,maxcol,X,y):
                print("Paramter")
                print("Paramter")
                print("Paramter")
                print(maxcol,end)
                Listing=default_model_regression(X,y,1,end,model_type)
                download_status=0
            else:

                error="Max Column Exceed"
        except Exception as ep:
            error="Wrong Prediction Columns"
        # y=le.fit_transform(df[pred_col]) 
    d={'data': df.to_html(),"listing":Listing ,"error":error,"download":download_status}
    
    return render(request,"default_regression.html",d)
def default_classification(request):
    Listing=[]
    
    error=""
    # Listing=pd.DataFrame()
    df = pd.DataFrame()
    if request.method == 'POST' :
        print("using post method")
        pred_col=request.POST["predcol"]
        model_type=request.POST["mtype"]
        end=request.POST["size1"]
        end=int(end)
        u=request.FILES['datafile']
        df=pd.read_csv(u)
        # print(df.head())
        X=df.drop(columns=pred_col)

        le=LabelEncoder()
        y=le.fit_transform(df[pred_col]) 
        Listing=default_model_classifier(X,y,1,end,model_type)
        
    
    d={'data': df.to_html(),"listing":Listing}
    print("listing")    
    print("listing")    
    # print(Listing)    
    return render(request,"default_classification.html",d)
def Extratreeclassification(request):
    parms={}
    Listing=[]
    error=""
    df = pd.DataFrame()
    ucriterion=["criterion"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
        print(type(parms))
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        maxdepth_value=request.POST.get("slidervalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("sample_split")
        sample_split_value=int(sample_split_value)
        # n_estimator_value=request.POST.get("n_estimator")
        # n_estimator_value=int(n_estimator_value)
        end=int(end)
        # ucriteion=request.POST["criteion"]
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('maxfeature')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        print(ucriterion,"slider value is:",)
        # *********************Creating Parameter************************
        ucriterion=ucriterion+criterion_list
        umaxfeature=umaxfeature+maxfeature_list
        for i in range(1,maxdepth_value,5):
                        max_depth.append(i)
        # for i in range(1,n_estimator_value,5):
        #                 n_estimator.append(i)
        for i in range(1,sample_split_value,1):
                        sample_split.append(i)
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
        creating_hyper_paramerter(parms,n_estimator)
        creating_hyper_paramerter(parms,sample_split)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        X=df.drop(columns=pred_col)

        le=LabelEncoder()
        y=le.fit_transform(df[pred_col]) 
        Listing=random_forest_classifier(X,y,1,end,parms)
    d={'data': df.to_html(),"listing":Listing}
    return render(request,"Extratreeclassification.html",d)
def random_forest_classification(request):
    parms={}
    Listing=[]
    download_status=0
    error=""
    df = pd.DataFrame()
    ucriterion=["criterion"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
        print(type(parms))
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        maxdepth_value=request.POST.get("slidervalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("sample_split")
        sample_split_value=int(sample_split_value)
        n_estimator_value=request.POST.get("n_estimator")
        n_estimator_value=int(n_estimator_value)
        end=int(end)
        # ucriteion=request.POST["criteion"]
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('maxfeature')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        print(ucriterion,"slider value is:",)
        # *********************Creating Parameter************************
        ucriterion=ucriterion+criterion_list
        umaxfeature=umaxfeature+maxfeature_list
        for i in range(1,maxdepth_value,5):
                        max_depth.append(i)
        for i in range(1,n_estimator_value,5):
                        n_estimator.append(i)
        for i in range(1,sample_split_value,1):
                        sample_split.append(i)
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
        creating_hyper_paramerter(parms,n_estimator)
        creating_hyper_paramerter(parms,sample_split)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        try:
        # print(df.head())
            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=RandomForestClassifier()
                Listing=all_classification_model(X,y,1,end,parms,model_name)
                # Listing=random_forest_classifier(X,y,1,end,parms)
                download_status=1
            else:
                    error="Max Column Exceed"
        except Exception as ep:
              print(ep)

        # print()
    d={'data': df.to_html(),"listing":Listing,"download":download_status}
    return render(request,"random_forest_clasification.html",d)
def interface_svc(request):
    parms={}
    Listing=[]
    download_status=0
    error=""
    df = pd.DataFrame()
    ukernal=["kernel"]
    ugamma=["gamma"]
    udegree=["degree"]
    # n_estimator=["n_estimators"]
    # sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
        print(type(parms))
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        degree_value=request.POST.get("degreevalue")
        degree_value=int(degree_value)
        # sample_split_value=request.POST.get("sample_split")
        # sample_split_value=int(sample_split_value)
        # n_estimator_value=request.POST.get("n_estimator")
        # n_estimator_value=int(n_estimator_value)
        # end=int(end)
        # ucriteion=request.POST["criteion"]
        kernal_list= request.POST.getlist('kernal')
        gamma_list= request.POST.getlist('gamma')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        # print(ucriterion,"slider value is:",)
        # *********************Creating Parameter************************
        ukernal=ukernal+kernal_list
        ugamma=ugamma+gamma_list
        # for i in range(1,maxdepth_value,5):
        #                 max_depth.append(i)
        # for i in range(1,n_estimator_value,5):
        #                 n_estimator.append(i)
        for i in range(1,degree_value,1):
                        udegree.append(i)
        creating_hyper_paramerter(parms,ugamma)
        creating_hyper_paramerter(parms,ukernal)
        creating_hyper_paramerter(parms,udegree)
        # creating_hyper_paramerter(parms,n_estimator)
        # creating_hyper_paramerter(parms,sample_split)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:
        # print(df.head())
            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=SVC()
                Listing=all_classification_model(X,y,1,end,parms,model_name)
            
                download_status=1
            else:
                    error="Max Column Exceed"
        except Exception as ep:
              print(ep)
        # print()
        
    d={'data': df.to_html(),"listing":Listing,"error":error ,"download":download_status}
    return render(request,"interface_svc.html",d)
def interface_gradientboosting_classifier(request):
    parms={}
    Listing=[]
    download_status=0
    error=""
    df = pd.DataFrame()
    ucriterion=["criterion"]
    ulearning_rate=["learning_rate"]
    uloss=["loss"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
        print(type(parms))
        print("inside post methid of  Gradient boosting class")
        print("inside post methid of  Gradient boosting class")
        print("inside post methid of Gradient boosting  class")
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        learing_rate_value=request.POST.get("learingvalue")
        learing_rate_value=float(learing_rate_value)
        maxdepth_value=request.POST.get("maxdepthvalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("minsamplevalue")
        sample_split_value=int(sample_split_value)
        n_estimator_value=request.POST.get("nestimatorsvalue")
        n_estimator_value=int(n_estimator_value)
        # end=int(end)
        # ucriteion=request.POST["criteion"]
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('max_features')
        loss_list= request.POST.getlist('loss')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        print(ucriterion,"slider value is:",)
        # *********************Creating Parameter************************
        ucriterion=ucriterion+criterion_list
        umaxfeature=umaxfeature+maxfeature_list
        uloss=uloss+loss_list      
        for i in range(1,maxdepth_value,5):
                        max_depth.append(i)
        for i in range(1,n_estimator_value,5):
                        n_estimator.append(i)
        for i in range(1,sample_split_value,1):
                        sample_split.append(i)
       
        for i in np.arange(1,learing_rate_value,0.2):
              ulearning_rate.append(round(i,1))
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
        creating_hyper_paramerter(parms,n_estimator)
        creating_hyper_paramerter(parms,sample_split)
        creating_hyper_paramerter(parms,ulearning_rate)
        creating_hyper_paramerter(parms,uloss)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:
        # print(df.head())
            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=GradientBoostingClassifier()
                Listing=all_classification_model(X,y,1,end,parms,model_name)
               
                download_status=1
            else:
                    error="Max Column Exceed"
        except Exception as ep:
              print(ep)
              error="Wrong Prediction Columns"
        # print()
    d={'data': df.to_html(),"listing":Listing,"error":error ,"download":download_status}
    
     
    return render (request,"gradientboosting_classifier.html",d)
def interface_decisiontree_classifier(request):
    parms={}
    Listing=[]
    download_status=0
    error=""
    df = pd.DataFrame()
    ucriterion=["criterion"]
 
    usplitter=["splitter"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    # n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
        print(type(parms))
        print("inside post methid of  Gradient boosting class")
        print("inside post methid of  Gradient boosting class")
        print("inside post methid of Gradient boosting  class")
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        # learing_rate_value=request.POST.get("learingvalue")
        # learing_rate_value=float(learing_rate_value)
        maxdepth_value=request.POST.get("maxdepthvalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("minsamplevalue")
        sample_split_value=int(sample_split_value)
        # n_estimator_value=request.POST.get("nestimatorsvalue")
        # n_estimator_value=int(n_estimator_value)
        # end=int(end)
        # ucriteion=request.POST["criteion"]
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('max_features')
        splitter_list= request.POST.getlist('splitter')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        print(ucriterion,"slider value is:",)
        # *********************Creating Parameter************************
        ucriterion=ucriterion+criterion_list
        umaxfeature=umaxfeature+maxfeature_list
        usplitter=usplitter+splitter_list      
        for i in range(1,maxdepth_value,5):
                        max_depth.append(i)
        # for i in range(1,n_estimator_value,5):
        #                 n_estimator.append(i)
        for i in range(1,sample_split_value,1):
                        sample_split.append(i)
       
        # for i in np.arange(1,learing_rate_value,0.2):
        #       ulearning_rate.append(round(i,1))
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
        # creating_hyper_paramerter(parms,n_estimator)
        creating_hyper_paramerter(parms,sample_split)
        # creating_hyper_paramerter(parms,ulearning_rate)
        creating_hyper_paramerter(parms,usplitter)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:
        # print(df.head())
            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=DecisionTreeClassifier()
                Listing=all_classification_model(X,y,1,end,parms,model_name)    
                download_status=1
            else:
                    error="Max Column Exceed"
        except Exception as ep:
              print(ep)
        # print()
    d={'data': df.to_html(),"listing":Listing,"error":error ,"download":download_status}
    return render(request,"decisiontree_classifier.html",d)
def interface_knn_classifier(request):
    parms={}
    Listing=[]
    download_status=0
    error=""
    df = pd.DataFrame()
    uweights=["weights"]
    un_neighbors=["n_neighbors"]
    ualgorithm=["algorithm"]
    # umulti_class=["multi_class"]
    # n_estimator=["n_estimators"]
    # sample_split=["min_samples_split"]
    if request.method == 'POST':
        print(type(parms))
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        weights_list=request.POST.getlist("weights")
        alogorith_list= request.POST.getlist('algorithm')
        n_neighnour_value=request.POST["kvalue"]
        # n_neighnour_value2=request.POST.get("sample_split")
        print("Print nneighbor")
        print("Print nneighbor")
        print("Print nneighbor")
        print(n_neighnour_value)
        n_neighnour_value=int(n_neighnour_value)
        # mullist= request.POST.getlist('multi_class')
     
        datafi=request.FILES['datafile']
        # *********************Creating Parameter************************
        uweights=uweights+weights_list
        ualgorithm=ualgorithm + alogorith_list
        # umulti_class=multi_class_list+umulti_class
        # print(penalty_list,C_value,solver_list,multi_class_list)
        for i in range(1,n_neighnour_value,1):
                        un_neighbors.append(i)
        # for i in range(1,n_estimator_value,5):
        #                 n_estimator.append(i)
        # for i in range(1,sample_split_value,1):
        #                 sample_split.append(i)
        creating_hyper_paramerter(parms,uweights)
        creating_hyper_paramerter(parms,ualgorithm)
        creating_hyper_paramerter(parms,un_neighbors)
        # creating_hyper_paramerter(parms,uc)
        # creating_hyper_paramerter(parms,sample_split)
        # print(type(parms))
        parms=dict(parms)
        # print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        try:
        # print(df.head())
            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=KNeighborsClassifier()
                Listing=all_classification_model(X,y,1,end,parms,model_name)
                # Listing=knn_classfier(X,y,1,end,parms)
                download_status=1
            else:
                    error="Max Column Exceed"
        except Exception as ep:
              print(ep)
    d={'data': df.to_html(),"listing":Listing,"error":error ,"download":download_status, "chosenparameter":parms}
    return render(request,"knn_classifier.html",d)

def extratreeregression(request):
    parms={}
    Listing=[]
    error=""
    df = pd.DataFrame()
    ucriterion=["criterion"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    # n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
        print(type(parms))
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        maxdepth_value=request.POST.get("slidervalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("sample_split")
        sample_split_value=int(sample_split_value)
        # n_estimator_value=request.POST.get("n_estimator")
        # n_estimator_value=int(n_estimator_value)
        # end=int(end)
        # ucriteion=request.POST["criteion"]
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('maxfeature')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        print(ucriterion,"slider value is:",)
        # *********************Creating Parameter************************
        ucriterion=ucriterion+criterion_list
        umaxfeature=umaxfeature+maxfeature_list
        for i in range(1,maxdepth_value,5):
                        max_depth.append(i)
        # for i in range(1,n_estimator_value,5):
        #                 n_estimator.append(i)
        for i in range(1,sample_split_value,1):
                        sample_split.append(i)
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
        # creating_hyper_paramerter(parms,n_estimator)
        creating_hyper_paramerter(parms,sample_split)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        try:
            df=pd.read_csv(datafi)
        except Exception as ep:
            error="wrong File"
        maxcol=df.shape[1]-2
        # print(df.head())
        try:
            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            if parameter_checkup(end,maxcol,X,y):
                model_name=ExtraTreeRegressor()
                Listing=all_regression(X,y,1,end,parms,model_name)
            else:
                error="Max Column Exceed"
        except Exception as ep:
            print("exception")
            print("exception")
            print("exception")
            print("exception")
            print("exception")
            print(parms)
            print(ep)
            error="Wrong Prediction Columns"

        # print()
    d={'data': df.to_html(),"listing":Listing,"error":error}
    return  render(request,"Extratreeregression.html",d)
def random_forest_regression(request):
    parms={}
    Listing=[]
    id_generator=0
    download_status=0
    error=""
    df = pd.DataFrame()
    ucriterion=["criterion"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
        print(type(parms))
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        maxdepth_value=request.POST.get("slidervalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("sample_split")
        sample_split_value=int(sample_split_value)
        n_estimator_value=request.POST.get("n_estimator")
        n_estimator_value=int(n_estimator_value)
        # end=int(end)
        # ucriteion=request.POST["criteion"]
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('maxfeature')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        print(ucriterion,"slider value is:",)
        # *********************Creating Parameter************************
        ucriterion=ucriterion+criterion_list
        umaxfeature=umaxfeature+maxfeature_list
        for i in range(1,maxdepth_value,5):
                        max_depth.append(i)
        for i in range(1,n_estimator_value,5):
                        n_estimator.append(i)
        for i in range(1,sample_split_value,1):
                        sample_split.append(i)
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
        creating_hyper_paramerter(parms,n_estimator)
        creating_hyper_paramerter(parms,sample_split)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=RandomForestRegressor()
                Listing,model=all_regression(X,y,1,end,parms,model_name)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                id_generator=model_saving(model,remaining_url,request.user.username)
                # Listing=random_forest_regression2(X,y,1,end,parms)
                download_status=1
                print("doownload status 1")
                print("doownload status 1")
                print("doownload status 1")
                print("doownload status 1")
                print("doownload status 1")
            else:
                error="Max Column Exceed"

        except Exception as ep:
            error="Wrong Prediction Columns"
            error="Wrong Prediction Columns"
            error="Wrong Prediction Columns"
            error="Wrong Prediction Columns"
            error="Wrong Prediction Columns"
            print(ep)
        # print()
    d={'data': df.to_html(),"listing":Listing,"error":error ,"download":download_status,"id_generator":id_generator}
    
    #     # print()
    # d={'data': df.to_html(),"listing":Listing}
    return render(request,"random_forest_regression.html",d)
# Create your views here.
def interface_svm(request):
    parms={}
    id_generator=0
    Listing=[]
    download_status=0
    error=""
    df = pd.DataFrame()
    ukernal=["kernel"]
    ugamma=["gamma"]
    udegree=["degree"]
    # n_estimator=["n_estimators"]
    # sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
        print(type(parms))
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        degree_value=request.POST.get("degreevalue")
        degree_value=int(degree_value)
        # sample_split_value=request.POST.get("sample_split")
        # sample_split_value=int(sample_split_value)
        # n_estimator_value=request.POST.get("n_estimator")
        # n_estimator_value=int(n_estimator_value)
        # end=int(end)
        # ucriteion=request.POST["criteion"]
        kernal_list= request.POST.getlist('kernal')
        gamma_list= request.POST.getlist('gamma')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        # print(ucriterion,"slider value is:",)
        # *********************Creating Parameter************************
        ukernal=ukernal+kernal_list
        ugamma=ugamma+gamma_list
        # for i in range(1,maxdepth_value,5):
        #                 max_depth.append(i)
        # for i in range(1,n_estimator_value,5):
        #                 n_estimator.append(i)
        for i in range(1,degree_value,1):
                        udegree.append(i)
        creating_hyper_paramerter(parms,ugamma)
        creating_hyper_paramerter(parms,ukernal)
        creating_hyper_paramerter(parms,udegree)
        # creating_hyper_paramerter(parms,n_estimator)
        # creating_hyper_paramerter(parms,sample_split)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
            #  Listing=svm_regression(X,y,1,end,parms)
                model_name=SVR()
                Listing,model=all_regression(X,y,1,end,parms,model_name)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                id_generator=model_saving(model,remaining_url,request.user.username)
                download_status=1
            else:
                error="Max Column Exceed"

        except Exception as ep:
            error="Wrong Prediction Columns"
            print(ep)
        # print()
    d={'data': df.to_html(),"listing":Listing,"error":error ,"download":download_status,"id_generator":id_generator}
    return render(request,"interface_svm.html",d)

def interface_gradientboosting_regressor(request):
    parms={}
    Listing=[]
    download_status=0
    id_generator=0
    error=""
    df = pd.DataFrame()
    ucriterion=["criterion"]
    ulearning_rate=["learning_rate"]
    uloss=["loss"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
        print(type(parms))
        print("inside post methid of  Gradient boosting class")
        print("inside post methid of  Gradient boosting class")
        print("inside post methid of Gradient boosting  class")
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        learing_rate_value=request.POST.get("learingvalue")
        learing_rate_value=float(learing_rate_value)
        maxdepth_value=request.POST.get("maxdepthvalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("minsamplevalue")
        sample_split_value=int(sample_split_value)
        n_estimator_value=request.POST.get("nestimatorsvalue")
        n_estimator_value=int(n_estimator_value)
        # end=int(end)
        # ucriteion=request.POST["criteion"]
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('max_features')
        loss_list= request.POST.getlist('loss')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        print(ucriterion,"slider value is:",)
        # *********************Creating Parameter************************
        ucriterion=ucriterion+criterion_list
        umaxfeature=umaxfeature+maxfeature_list
        uloss=uloss+loss_list      
        for i in range(1,maxdepth_value,5):
                        max_depth.append(i)
        for i in range(1,n_estimator_value,5):
                        n_estimator.append(i)
        for i in range(2,sample_split_value,1):
                        sample_split.append(i)
       
        for i in np.arange(1,learing_rate_value,0.2):
              ulearning_rate.append(round(i,1))
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
        creating_hyper_paramerter(parms,n_estimator)
        creating_hyper_paramerter(parms,sample_split)
        creating_hyper_paramerter(parms,ulearning_rate)
        creating_hyper_paramerter(parms,uloss)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=GradientBoostingRegressor()
                Listing,model=all_regression(X,y,1,end,parms,model_name)
                # print(type(model))
                # print("model is",model)     
                # print("************************************")
                # print(datafi)
                # print(f"The User id is {(request.user.id)}")
                # print(f"The User id is {(request.user.username)}")
                # print(f"The User id is {model_name}")
                # print(f"The User id is{pred_col[:3]}")
                
                # print(f"model name will be {(str(model_name))[1:-2]+request.user.username[:3]+str(pred_col[:3])+str(end)+str(request.user.id)}")            
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                id_generator=model_saving(model,remaining_url,request.user.username)
                # print(type(remaining_url))
                # base_url="media/"
                # full_url=base_url+remaining_url
                # print(type(full_url))
                # id_generator=random.randint(1000,9999)
                # print(id_generator)
                # # pickle.dump(model, open(full_url, 'wb'))
                # joblib.dump(model, open(full_url, 'wb'))
                # # *******************Saving in database************************
                # result=TrainedModel(username=request.user.username,location=full_url,modelname=remaining_url,uniqueid=id_generator)
                # result.save()
                print("WOKING IN PICKLE")
                # print("WOKING IN JOBLIB")
                # print("WOKING IN JOBLIB")
            #  Listing=gradientboosting_regression(X,y,1,end,parms)
                download_status=1
            else:
                error="Max Column Exceed"

        except Exception as ep:
            error="Wrong Prediction Columns"
            print(ep)
        # print()
    d={'data': df.to_html(),"listing":Listing,"error":error ,"download":download_status, "id_generator":id_generator}
    
     
    return render (request,"gradientboosting_regressor.html",d)
def interface_decisiontree_regressor(request):
    parms={}
    Listing=[]
    id_generator=0
    download_status=0
    error=""
    df = pd.DataFrame()
    ucriterion=["criterion"]
 
    usplitter=["splitter"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    # n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
        print(type(parms))
        print("inside post methid of  Gradient boosting class")
        print("inside post methid of  Gradient boosting class")
        print("inside post methid of Gradient boosting  class")
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        # learing_rate_value=request.POST.get("learingvalue")
        # learing_rate_value=float(learing_rate_value)
        maxdepth_value=request.POST.get("maxdepthvalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("minsamplevalue")
        sample_split_value=int(sample_split_value)
        # n_estimator_value=request.POST.get("nestimatorsvalue")
        # n_estimator_value=int(n_estimator_value)
        # end=int(end)
        # ucriteion=request.POST["criteion"]
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('max_features')
        splitter_list= request.POST.getlist('splitter')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        print(ucriterion,"slider value is:",)
        # *********************Creating Parameter************************
        ucriterion=ucriterion+criterion_list
        umaxfeature=umaxfeature+maxfeature_list
        usplitter=usplitter+splitter_list      
        for i in range(1,maxdepth_value,5):
                        max_depth.append(i)
        # for i in range(1,n_estimator_value,5):
        #                 n_estimator.append(i)
        for i in range(1,sample_split_value,1):
                        sample_split.append(i)
       
        # for i in np.arange(1,learing_rate_value,0.2):
        #       ulearning_rate.append(round(i,1))
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
        # creating_hyper_paramerter(parms,n_estimator)
        creating_hyper_paramerter(parms,sample_split)
        # creating_hyper_paramerter(parms,ulearning_rate)
        creating_hyper_paramerter(parms,usplitter)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=DecisionTreeRegressor()
                Listing,model=all_regression(X,y,1,end,parms,model_name)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                id_generator=model_saving(model,remaining_url,request.user.username)
                # Listing=decisiontree_regressor(X,y,1,end,parms)
                download_status=1
            else:
                error="Max Column Exceed"

        except Exception as ep:
            error="Wrong Prediction Columns"
        # print()
    d={'data': df.to_html(),"listing":Listing,"error":error ,"download":download_status, "id_generator":id_generator}
    return render(request,"decisiontree_regressor.html",d)
def signup(request):
    if request.method == 'POST':
        first_name=request.POST['uname']
        lastname=request.POST['lname']
        email=request.POST['emailid']
        password=request.POST['passwrd']
        print("printing Signup deteial")
        print(first_name,lastname,email)
        print(password)
        try:
            user=User.objects.create_user(first_name=first_name,last_name=lastname,username=email,password=password)
            Signup.objects.create(user=user)
            error="no"
        except Exception as ep:
            print(ep)
            error="yes"
    print("no in if condiotn")
    return render(request,"signup.html")
def user_trained_model(request):
     data=TrainedModel.objects.filter(username=request.user.username)
     print(f"the username is {request.user.username}")
     print("Printing Trained Model")
    #  print(data)
     d={'data': data}
     return render(request,"user_trained_model.html",d)

def interface_download_random_forest_regressor(request):
    download_status=0
    parameter_list=[]
    Listing=[]
    error=""
    df = pd.DataFrame()
    ucriterion=["criterion"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        maxdepth_value=request.POST["slidervalue"] #mnot
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST["sample_split"]
        sample_split_value=int(sample_split_value)
        n_estimator_value=request.POST["n_estimator"]
        n_estimator_value=int(n_estimator_value)
        end=int(end)
        # ucriteion=request.POST["criteion"]
        criterion_list= request.POST['criteion']
        maxfeature_list= request.POST['maxfeature']
        datafi=request.FILES['datafile']
        print(pred_col,end)
   
        parameter_list.append( end)
        parameter_list.append(criterion_list)
        parameter_list.append(maxfeature_list)
        parameter_list.append( sample_split_value)
        parameter_list.append(n_estimator_value)
        parameter_list.append(maxdepth_value)
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
             Listing=download_random_forest_regressor(X,y,parameter_list)
             request.session['trp'] = str(Listing)
             request.session['tp'] = 20
             print(Listing)
             
            #  request.session['trp'] = Listing

             download_status=1
            
            else:
                error="Max Column Exceed"

        except Exception as ep:
            error="Wrong Prediction Columns"
            print("etro")
            print(ep)
            print(ep)
            print(ep)
            print(ep)
        # print()
    d={'data': df.to_html(),"error":error ,"download":download_status,"model":Listing}
      
    return render (request,"download_random_forest_regressor.html",d)

def interface_download_logistic_regression(request):
    download_status=0
    parameter_list=[]
    Listing=[]
    error=""
    df = pd.DataFrame()
    # upenalty=["penalty"]
    # umulti_class=["multi_class"]
    # usolver=["solver"]
    # ucvalue=["cvalue"]
    # sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
      
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")

        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])

        penalty_list=request.POST["penalty"] #mnot
       
        umulti_class_list=request.POST["multi_class"]
       
        solver_list=request.POST["solver"]
       
        # ucriteion=request.POST["criteion"]
        c_value= int(request.POST['cvalue'])
       
        datafi=request.FILES['datafile']
        print(pred_col,end)
   
        parameter_list.append( end)
        parameter_list.append(penalty_list)
        parameter_list.append(umulti_class_list)
        parameter_list.append( solver_list)
        parameter_list.append(c_value)
        # parameter_list.append(maxdepth_value)
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
             Listing=download_logistic_regresion(X,y,parameter_list)
             print(Listing)
            #  request.session['trp'] = Listing

             download_status=1
            
            else:
                error="Max Column Exceed"

        except Exception as ep:
            error="Wrong Prediction Columns"
            print("etro")
            print(ep)
            print(ep)
            print(ep)
            print(ep)
        # print()
    d={'data': df.to_html(),"error":error ,"download":download_status}
    return render(request,"download_logistic_regression.html",d)

def interface_download_extreetree_regression(request):
    download_status=0
    parameter_list=[]
    Listing=[]
    error=""
    df = pd.DataFrame()
    # sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
      
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")

        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        criteion_list=request.POST["criteion"] #mnot
       
        maxfeature_list=request.POST["maxfeature"]
       
        # solver_list=request.POST["solver"]
       
        # ucriteion=request.POST["criteion"]
        ndepth_value= int(request.POST['ndepth'])
        sample_split_value= int(request.POST['sample_split'])
       
        datafi=request.FILES['datafile']
        print(pred_col,end)
   
        parameter_list.append( end)
        parameter_list.append(criteion_list)
        parameter_list.append(maxfeature_list)
        parameter_list.append(ndepth_value)
        parameter_list.append(sample_split_value)
        # parameter_list.append(maxdepth_value)
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
             Listing=download_extratree_regresion(X,y,parameter_list)
             pkl_data = pickle.dumps(Listing,"pra1")
             print("type")
             print("type")
             print(type(Listing))
             print(Listing)
             print("type")
             print("type")
             print(type(Listing))
            #  request.session['trp'] = Listing

             download_status=1
            
            else:
                error="Max Column Exceed"

        except Exception as ep:
            error="Wrong Prediction Columns"
            print("etro")
            print(ep)
            print(ep)
            print(ep)
            print(ep)
        # print()
    d={'data': df.to_html(),"error":error ,"download":download_status, "model":Listing}
      
    return render(request,"download_extra_tree_regression.html",d)
def interface_download_svr(request):
    download_status=0
    parameter_list=[]
    Listing=[]
    error=""
    df = pd.DataFrame()
    # sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
      
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")

        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        ukernal=request.POST["kernal"] #mnot
       
        ugamma=request.POST["gamma"]
       
        # solver_list=request.POST["solver"]
       
        # ucriteion=request.POST["criteion"]
        degreevalue= int(request.POST['degreevalue'])
  
       
        datafi=request.FILES['datafile']
        print(pred_col,end)
   
        parameter_list.append( end)
        parameter_list.append(ukernal)
        parameter_list.append(ugamma)
        parameter_list.append(degreevalue)
        # parameter_list.append(sample_split_value)
        # parameter_list.append(maxdepth_value)
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
             Listing=download_svr_regression(X,y,parameter_list)
            #  pkl_data = pickle.dumps(Listing,"pra1")
             print("type")
             print("type")
             print(type(Listing))
             print(Listing)
             print("type")
             print("type")
             print(type(Listing))
            #  request.session['trp'] = Listing

             download_status=1
            
            else:
                error="Max Column Exceed"

        except Exception as ep:
            error="Wrong Prediction Columns"
            print("etro")
            print(ep)
            print(ep)
            print(ep)
            print(ep)
        # print()
    d={'data': df.to_html(),"error":error ,"download":download_status, "model":Listing}

    return render(request,"download_svr_regression.html",d)
def interface_download_knn_regression(request):
    download_status=0
    parameter_list=[]
    Listing=[]
    error=""
    df = pd.DataFrame()
    # sample_split=["min_samples_split"]
    if request.method == 'POST':
        # print(type(parms))
      
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")

        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        uweight=request.POST["weights"] #mnot
       
        ualgorith=request.POST["algorithm"]
       
        # solver_list=request.POST["solver"]
       
        # ucriteion=request.POST["criteion"]
        kvalue= int(request.POST['kvalue'])
  
       
        datafi=request.FILES['datafile']
        print(pred_col,end)
   
        parameter_list.append( end)
        parameter_list.append(uweight)
        parameter_list.append(ualgorith)
        parameter_list.append(kvalue)
        # parameter_list.append(sample_split_value)
        # parameter_list.append(maxdepth_value)
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
             Listing=download_knn_regression(X,y,parameter_list)
            #  pkl_data = pickle.dumps(Listing,"pra1")
             print("type")
             print("type")
             print(type(Listing))
             print(Listing)
             print("type")
             print("type")
             print(type(Listing))
            #  request.session['trp'] = Listing

             download_status=1
            
            else:
                error="Max Column Exceed"

        except Exception as ep:
            error="Wrong Prediction Columns"
            print("etro")
            print(ep)
            print(ep)
            print(ep)
            print(ep)
        # print()
    d={'data': df.to_html(),"error":error ,"download":download_status, "model":Listing}
    return render(request,"download_knn_regression.html",d)
def rough_regression(request):
     return render(request,"rough_regression.html")

def download_model(request,id):
        print("****************************")
        print("download model id")
        print(id)
        data=TrainedModel.objects.filter(username=request.user.username,uniqueid=id).values_list('modelname', flat=True)
        print(f"the username is {request.user.username}")
        print("Printing Trained Model")
        base_url="D:/machine learning/Project/Website/Train_your_model/media/"
        remaining_url=data[0]
        file_path = base_url+remaining_url
        print(f"The full path is {file_path}")
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/force-download")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
def delete_model(request,id):
     print(f"the feletering {id}")
     data=TrainedModel.objects.filter(username=request.user.username,uniqueid=id).delete()
    #  print(data)
     return redirect("user_trained_model")
     


def interface_logistic_regression(request):
    parms={}
    Listing=[]
    download_status=0
    error=""
    df = pd.DataFrame()
    upenalty=["penalty"]
    uc=["C"]
    usolver=["solver"]
    umulti_class=["multi_class"]
    # n_estimator=["n_estimators"]
    # sample_split=["min_samples_split"]
    if request.method == 'POST':
        print(type(parms))
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        
        penalty_list=request.POST.getlist("penalty")
        C_value=int(request.POST.get("cvalue"))
        solver_list= request.POST.getlist('solver')
        multi_class_list= request.POST.getlist('multi_class')
     
        datafi=request.FILES['datafile']
        # *********************Creating Parameter************************
        upenalty=upenalty+penalty_list
        usolver=usolver+solver_list
        umulti_class=umulti_class+multi_class_list
        print(penalty_list,C_value,solver_list,multi_class_list)
        for i in range(1,C_value,1):
                        uc.append(i)
        # for i in range(1,n_estimator_value,5):
        #                 n_estimator.append(i)
        # for i in range(1,sample_split_value,1):
        #                 sample_split.append(i)
        creating_hyper_paramerter(parms,upenalty)
        creating_hyper_paramerter(parms,usolver)
        creating_hyper_paramerter(parms,umulti_class)
        creating_hyper_paramerter(parms,uc)
        # creating_hyper_paramerter(parms,sample_split)
        # print(type(parms))
        parms=dict(parms)
        # print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)

        try:
        # print(df.head())
            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                Listing=logistic_regression(X,y,1,end,parms)
                download_status=1
            else:
                    error="Max Column Exceed"
        except Exception as ep:
              print(ep)
        # print()
    d={'data': df.to_html(),"listing":Listing,"error":error ,"download":download_status, "chosendparameter":parms}
    return render(request,"Logistic_regression.html",d)

def interface_Knn_regressor(request):
    parms={}
    Listing=[]
    id_generator=0
    download_status=0
    error=""
    df = pd.DataFrame()
    uweights=["weights"]
    un_neighbors=["n_neighbors"]
    ualgorithm=["algorithm"]
    # umulti_class=["multi_class"]
    # n_estimator=["n_estimators"]
    # sample_split=["min_samples_split"]
    if request.method == 'POST':
        print(type(parms))
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        print("inside post methid of randomforest class")
        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        weights_list=request.POST.getlist("weights")
        alogorith_list= request.POST.getlist('algorithm')
        n_neighnour_value=request.POST["kvalue"]
        # n_neighnour_value2=request.POST.get("sample_split")
        print("Print nneighbor")
        print("Print nneighbor")
        print("Print nneighbor")
        print(n_neighnour_value)
        n_neighnour_value=int(n_neighnour_value)
        # mullist= request.POST.getlist('multi_class')
     
        datafi=request.FILES['datafile']
        # *********************Creating Parameter************************
        uweights=uweights+weights_list
        ualgorithm=ualgorithm + alogorith_list
        # umulti_class=multi_class_list+umulti_class
        # print(penalty_list,C_value,solver_list,multi_class_list)
        for i in range(1,n_neighnour_value,1):
                        un_neighbors.append(i)
        # for i in range(1,n_estimator_value,5):
        #                 n_estimator.append(i)
        # for i in range(1,sample_split_value,1):
        #                 sample_split.append(i)
        creating_hyper_paramerter(parms,uweights)
        creating_hyper_paramerter(parms,ualgorithm)
        creating_hyper_paramerter(parms,un_neighbors)
        # creating_hyper_paramerter(parms,uc)
        # creating_hyper_paramerter(parms,sample_split)
        # print(type(parms))
        parms=dict(parms)
        # print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        try:

            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=KNeighborsRegressor()
                print("hyperparameter")
                print("hyperparameter")
                print("hyperparameter")
                print("hyperparameter")
                print(parms)
                Listing,model=all_regression(X,y,1,end,parms,model_name)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                id_generator=model_saving(model,remaining_url,request.user.username)
                download_status=1
            else:
                    error="Max Column Exceed"

        except Exception as ep:
                print("eroor ocusing in running")
                print("eroor ocusing in running")
                print(ep)
                error="Wrong Prediction Columns"

        # print()
    d={'data': df.to_html(),"listing":Listing,"error":error ,"download":download_status, "chosenparameter":parms,"id_generator":id_generator}
    return render(request,"knn_regressor.html",d)


def interface_download_dataset(request):
      return render(request,"download_dataset.html")

def user_download_dataset(request):
     return render(request,"user_download_dataset.html")

def mobilephone_flipkart(request):
    print("Under Mobile Phone")
    # print(pagenumber)
    url_template = "https://www.flipkart.com/search?q=mobile&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&page={}"

    # Define the number of pages to scrape
    num_pages = request.GET.get('page', 1)
    num_pages=int(num_pages)
    try:
        for page_num in range(1, num_pages+1):
            # Create the URL for the current page
            url = url_template.format(page_num)

            response = requests.get(url)

            soup = BeautifulSoup(response.content, 'html.parser')

            results = soup.find_all('div', {'class': '_2kHMtA'})

            for result in results:
                # Extract brand name
                try:

                    brand = result.find('div', {'class': '_4rR01T'}).text
                except:
                    brand="not found"
                try:
                # Extract price
                    price = result.find('div', {'class': '_30jeq3 _1_WHN1'}).text
                except:
                    price="not found"
                
                # Extract memory, display, battery, and warranty
                features = result.find_all('ul', {'class': '_1xgFaf'})[0]
                feature_list = features.find_all('li')
                try:
                    memory = feature_list[0].text
                except:
                    memory="not found"

                try:
                    display = feature_list[1].text
                except:
                    display="not found"
                try:
                    
                    camera = feature_list[2].text
                except:
                    camera="not found"
                try:
                    battery_processor = feature_list[3].text
                
                    if battery_processor[0].isdigit():
                        battery=battery_processor
                        print(battery)
                    else:
                        battery="not found"
                except:
                    battery="not found"
        

                # Append the extracted data to the list
                # mobile_data.append({
                #     "Brand": brand,
                #     "Price": price,
                #     "Memory": memory,
                #     "Display": display,
                #     "Camera": camera,
                #     "battery ": battery,
                # })
                m = FlipkartMobileModel(brand=brand, price=price[1:], memory=memory, display=display, camera=camera, battery=battery)
                m.save()
                print('price: ', price[1:])
        messages.success(request, 'Data Extracted Successfully')
       
    except Exception as ep:
         messages.error(request, 'Data Extraction Failed')
         print(ep)
    # pkl_data = pickle.dumps(mobile_data)
    # response = HttpResponse(pkl_data, content_type='application/octet-stream')
    # response['Content-Disposition'] = 'attachment; filename="data.csv"'
        # pkl_url = request.build_absolute_uri(
    # for mobile in mobile_data:
    #     print(mobile)
    
    return redirect("interface_download_dataset") 


def retrivemobilephone_flipkart(request):
        data=FlipkartMobileModel.objects.all()
        print("Printing retive data")        
        # students = Student.objects.all()
        response = HttpResponse('text/csv')
        
        response['Content-Disposition'] = 'attachment; filename=students.csv'
        writer = csv.writer(response)
        writer.writerow(['BRAND', 'PRICE', 'MEMORY', 'DISPLAY', 'CAMERA', 'BATTERY'])
       
        studs = data.values_list('brand', 'price', 'memory', 'display', 'camera', 'battery')
        for std in studs:
            writer.writerow(std)
        return response
def retrive_specific_mobilephone_flipkart(request):
        record = request.GET.get('page', 1)
        record=int(record)
        print(record)
        data=FlipkartMobileModel.objects.all()[:record-1]
        print("Printing retive data")        
        # students = Student.objects.all()
        response = HttpResponse('text/csv')
        
        response['Content-Disposition'] = 'attachment; filename=students.csv'
        writer = csv.writer(response)
        writer.writerow(['BRAND', 'PRICE', 'MEMORY', 'DISPLAY', 'CAMERA', 'BATTERY'])
       
        studs = data.values_list('brand', 'price', 'memory', 'display', 'camera', 'battery')
        for std in studs:
            writer.writerow(std)
        return response
     
    #   return redirect("interface_download_dataset")
def delete_mobilephone_flipkart(request):
      try:
        data=FlipkartMobileModel.objects.all().delete()
        messages.success(request, 'Data Deleted Successfully')
      except:
           messages.error(request, 'Data Not Deleted')

      return redirect("interface_download_dataset")
def laptop_flipkart(request):
    num_pages = request.GET.get('page', 1)
    num_pages=int(num_pages)
    # URL of the website to be scraped
    url = 'https://www.flipkart.com/search?q=laptop&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off'

    # Initialize an empty list to store the laptop details
 

    try:
        for page in range(1, num_pages+1):
            # Send a request to the URL
            response = requests.get(url + '&page=' + str(page))

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all the laptops on the page
            laptops = soup.find_all('div', {'class': '_2kHMtA'})

            # Loop through each laptop and extract the required details
            for laptop in laptops:
                # Find the name of the laptop
                try:
                    
                    name = laptop.find('div', {'class': '_4rR01T'}).text
                except:
                    name = 'Not Found'
                try:
                # Find the processor of the laptop
                    processor = laptop.find('ul', {'class': '_1xgFaf'}).find_all('li')[0].text
                except:
                    processor = 'Not Found'
                try:
                # Find the RAM of the laptop
                    ram = laptop.find('ul', {'class': '_1xgFaf'}).find_all('li')[1].text
                except:
                    ram = 'Not Found'

                # Find the operating system of the laptop
                try:
                    os = laptop.find('ul', {'class': '_1xgFaf'}).find_all('li')[2].text
                except:
                    os = 'Not Found'
                try:
                        
                # Find the hard disk capacity of the laptop
                    hdd = laptop.find('ul', {'class': '_1xgFaf'}).find_all('li')[3].text
                except:
                    hdd = 'Not Found'
                try:
                # Find the display size of the laptop
                    display = laptop.find('ul', {'class': '_1xgFaf'}).find_all('li')[4].text
                except:
                    display = 'Not Found'
                # Find the warranty of the laptop
                # warranty = laptop.find('ul', {'class': '_1xgFaf'}).find_all('li')[5].text
                try:
                # Find the price of the laptop
                    price = laptop.find('div', {'class': '_30jeq3 _1_WHN1'}).text
                except:
                    price = 'Not Found'
                # laptops_list.append({
                #     "name":name,
                #     "Processor":processor,
                #     "RAM":ram,
                #     "Operating System":os,
                #     "Hard Disk":hdd,
                #     "Dsiplay Size":display,
                #     "price":price[1:]
                # })
                result = FlipkartLaptopModel(name=name, processor=processor, ram=ram, opearting_system=os, hard_disk=hdd, display=display, price=price[1:])
                result.save()
        messages.success(request, 'Data Extracted Successfully')
    except:
        messages.error(request, "Data Not Extracted")
    
    return redirect("interface_download_dataset") 
def retrivelaptop_flipkart(request):
        data=FlipkartLaptopModel.objects.all()
        print("Printing retive data")        
        # students = Student.objects.all()
        response = HttpResponse('text/csv')
        
        response['Content-Disposition'] = 'attachment; filename=students.csv'
        writer = csv.writer(response)
        # writer.writerow(['BRAND', 'PRICE', 'MEMORY', 'DISPLAY', 'CAMERA', 'BATTERY'])
        writer.writerow(['NAME', 'PROCESSOR', 'RAM', 'OPERATING SYSTEM', 'HARD DISK', 'DISPLAY', 'PRICE'])
       
        # studs = data.values_list('brand', 'price', 'memory', 'display', 'camera', 'battery')
        studs = data.values_list('name', 'processor', 'ram', 'opearting_system', 'hard_disk', 'display', 'price')
        for std in studs:
            writer.writerow(std)
        return response

def retrive_specific_laptop_flipkart(request):
        record = request.GET.get('page', 1)
        record=int(record)
        print(record)
        data=FlipkartLaptopModel.objects.all()[:record-1]
        print("Printing retive data")        
        # students = Student.objects.all()
        response = HttpResponse('text/csv')
        
        response['Content-Disposition'] = 'attachment; filename=students.csv'
        writer = csv.writer(response)
        # writer.writerow(['BRAND', 'PRICE', 'MEMORY', 'DISPLAY', 'CAMERA', 'BATTERY'])
        writer.writerow(['NAME', 'PROCESSOR', 'RAM', 'OPERATING SYSTEM', 'HARD DISK', 'DISPLAY', 'PRICE'])
       
        # studs = data.values_list('brand', 'price', 'memory', 'display', 'camera', 'battery')
        studs=data.values_list('name', 'processor', 'ram', 'opearting_system', 'hard_disk', 'display', 'price')
        for std in studs:
            writer.writerow(std)
        return response
     
def delete_laptop_flipkart(request):
      try:
        data=FlipkartLaptopModel.objects.all().delete()
        messages.success(request, 'Data Deleted Successfully')
      except:
           messages.error(request, "No data to delete")
      return redirect("interface_download_dataset")

def interface_television_flipkart(request):

    url = "https://www.flipkart.com/search?q=telivsion&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"


    num_pages = request.GET.get('page', 1)
    num_pages=int(num_pages)
    print((num_pages))
    try:
        for i in range(1, num_pages+1):
            url_page = url + "&page=" + str(i)
           
            
            response = requests.get(url_page)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all the products on the page
            try:
                products = soup.find_all("div", {"class": "_3pLy-c row"})

            # Loop through each product and extract the required information
                for product in products:
                    try:
                        # Extract the model name
                        model = product.find("div", {"class": "_4rR01T"}).text
                    except:
                        model = "NA"
                    try:

                        # Extract the operating system
                        os = product.find("ul", {"class": "_1xgFaf"}).li.text
                    except:
                        os = "NA"
                    
                    try:
                        rating = product.find("div", {"class": "_3LWZlK"}).text
                    except:
                        rating = "NA"
                    try:
                        display = product.find("ul", {"class": "_1xgFaf"}).find_all("li")[1].text
                    except:
                        display = "NA"
                    try:
                        warranty = product.find("ul", {"class": "_1xgFaf"}).find_all("li")[2].text
                    except:
                        warranty = "NA"
                    try:
                        price = product.find("div", {"class": "_30jeq3 _1_WHN1"}).text
                    except:
                        price = "NA"
                    
                    result=FlipkartTelivisionModel(name=model, operating_system =os, rating=rating, display=display, warrently=warranty, price=price[1:])
                    result.save()
                  
            except:
                    pass
        messages.success(request, 'Data Extracted Successfully')

    except Exception as ep:
        print(ep)       
        messages.error(request, 'Data Extracted Successfully')
            
    return redirect("interface_download_dataset")

def retrive_television_flipkart(request):
        data=FlipkartTelivisionModel.objects.all()
        print("Printing retive data")        
        # students = Student.objects.all()
        response = HttpResponse('text/csv')
        
        response['Content-Disposition'] = 'attachment; filename=students.csv'
        writer = csv.writer(response)
     
        writer.writerow(['NAME', 'OPERATING SYSTEM', 'RATING', 'DISPLAY', 'WARRANTY', 'PRICE'])
       
        # studs = data.values_list('brand', 'price', 'memory', 'display', 'camera', 'battery')
        # studs = data.values_list('name', 'processor', 'ram', 'opearting_system', 'hard_disk', 'display', 'price')
        studs=data.values_list('name', 'operating_system', 'rating', 'display', 'warrently', 'price')
        for std in studs:
            writer.writerow(std)
        return response
def retrive_specific_telivision_flipkart(request):
        record = request.GET.get('page', 1)
        record=int(record)
        print(record)
        data=FlipkartTelivisionModel.objects.all()[:record-1]
        print("Printing retive data")        
        # students = Student.objects.all()
        response = HttpResponse('text/csv')
        
        response['Content-Disposition'] = 'attachment; filename=students.csv'
        writer = csv.writer(response)
        # writer.writerow(['BRAND', 'PRICE', 'MEMORY', 'DISPLAY', 'CAMERA', 'BATTERY'])
        writer.writerow(['NAME', 'OPERATING SYSTEM', 'RATING', 'DISPLAY', 'WARRANTY', 'PRICE'])
       
        # studs = data.values_list('brand', 'price', 'memory', 'display', 'camera', 'battery')
        studs = data.values_list('name', 'operating_system', 'rating', 'display', 'warrently', 'price')
        for std in studs:
            writer.writerow(std)
        return response
     
def delete_television_flipkart(request):
      try:
           
        data= FlipkartTelivisionModel.objects.all().delete()
        messages.success(request, 'Data Deleted Successfully')
      except:
           messages.error(request, 'Data Not Deleted')
      return redirect("interface_download_dataset")

def earphone_flipkart(request):
      
      base_url = "https://www.flipkart.com/search?q=earphone&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&page="
      num_pages = request.GET.get('page', 1)
      num_pages=int(num_pages)
      print(num_pages)
      headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
   
    # Loop through each page of earphone search results
      for page in range(1, num_pages+1):
    # Construct the URL for the current page
        url = base_url + str(page)

        # Send a GET request to the URL
        response = requests.get(url, headers=headers)

        # Create a BeautifulSoup object by parsing the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all the links to the individual earphone product pages
        try:
            product_links = soup.find_all('a', class_='s1Q9rs')
          

            # Loop through each product link and scrape its details
            for link in product_links:
                # Extract the URL of the product page
                product_url = "https://www.flipkart.com" + link.get('href')

                # Send a GET request to the product URL
                product_response = requests.get(product_url, headers=headers)

                # Create a BeautifulSoup object by parsing the HTML content of the product page
                product_soup = BeautifulSoup(product_response.content, 'html.parser')

                # Find the general details for the earphonep
                model_detail = product_soup.find_all('li', class_='_21lJbe')
                try:
                    model_name=model_detail[0].get_text()
                except:
                    model_name="Not Available"
                try:

                    color=model_detail[1].get_text()
                except:
                    color="Not Available"
                try:

                    headphone_type=model_detail[2].get_text()
                except:
                    headphone_type="Not Available"
                try:

                    inline_remote=model_detail[3].get_text()
                except:
                    inline_remote="Not Available"
                try:

                    connectivity=model_detail[5].get_text()
                except:
                    connectivity="Not Available"
                try:

                    price = product_soup.find('div', class_='_30jeq3 _16Jk6d').get_text()
                except:
                    price="Not Available"
                
                print(model_name, color, headphone_type, inline_remote, connectivity, price)
                result=FlipkartEarphoneModel(name=model_name, color=color, headphone_type=headphone_type, inline_remote=inline_remote, connectivity=connectivity, price=price[1:])
                result.save()
             
            messages.success(request, 'Data Saved Successfully')
        except:
            messages.error(request, 'Data Not Saved')
        return redirect("interface_download_dataset")
    
def retrive_earphone_flipkart(request):
        data=FlipkartEarphoneModel.objects.all()
        print("Printing retive data")        
        # students = Student.objects.all()
        response = HttpResponse('text/csv')
        
        response['Content-Disposition'] = 'attachment; filename=students.csv'
        writer = csv.writer(response)
     
        writer.writerow(['NAME', 'COLOR', 'HEADPHONE TYPE', 'INLINE REMOTE', 'CONNECTIVITY', 'PRICE'])
       
    
        studs=data.values_list('name', 'color', 'headphone_type', 'inline_remote', 'connectivity', 'price')
        for std in studs:
            writer.writerow(std)
        return response
def retrive_specific_earphone_flipkart(request):
        record = request.GET.get('page', 1)
        record=int(record)
        print(record)
        data=FlipkartEarphoneModel.objects.all()[:record-1]
        print("Printing retive data")        
        # students = Student.objects.all()
        response = HttpResponse('text/csv')
        
        response['Content-Disposition'] = 'attachment; filename=students.csv'
        writer = csv.writer(response)
        # writer.writerow(['BRAND', 'PRICE', 'MEMORY', 'DISPLAY', 'CAMERA', 'BATTERY'])
        wti = ['NAME', 'COLOR', 'HEADPHONE TYPE', 'INLINE REMOTE', 'CONNECTIVITY', 'PRICE']
       
        # studs = data.values_list('brand', 'price', 'memory', 'display', 'camera', 'battery')
        studs = data.values_list('name', 'color', 'headphone_type', 'inline_remote', 'connectivity', 'price')
        for std in studs:
            writer.writerow(std)
        return response
     
def delete_earphone_flipkart(request):
      try:
           
        data= FlipkartEarphoneModel.objects.all().delete()
        messages.success(request, "Data Deleted Successfully")
      except:
           messages.error(request, "Data Not Deleted")
      return redirect("interface_download_dataset")

def bike_flipkart(request):
    bike_data = []
    num_pages = request.GET.get('page', 1)
    num_pages=int(num_pages)
# Define the base URL, the number of pages to scrape, and the headers for the GET requests
    base_url = "https://www.autox.com/new-bike-launches-in-india/page"
    print(num_pages)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    # Loop through each page of bike launches
    try:
        for page in range(1, num_pages+1):
                # Construct the URL for the current page
            url = f"{base_url}/{page}"

            # Send a GET request to the URL
            response = requests.get(url, headers=headers)

            # Create a BeautifulSoup object by parsing the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all the links to the individual bike launch articles\
            article_links = soup.find_all('a', class_='model-title')

            # Loop through each article link and scrape its details
            for link in article_links:
                # Extract the URL of the article
                article_url = link.get('href')

                # Send a GET request to the article URL
                article_response = requests.get(article_url, headers=headers)

                # Create a BeautifulSoup object by parsing the HTML content of the article
                article_soup = BeautifulSoup(article_response.content, 'html.parser')

                # Find the model name, price, variant, displacement, max power, and mileage details from the article
                try:
                    model_name = article_soup.find('h1', class_='model-page-title').get_text()
                except:
                    model_name = "Not Available"
                try:
                    price = article_soup.find('span', class_='price').get_text()
                except:
                    price = "Not Available"
                # variant = article_soup.find('div', text='select_box').find_next_sibling('div').get_text()
                try:
                    displacement = article_soup.find('td', text='Displacement').find_next_sibling('td').get_text()
                except:
                    displacement = "Not Available"
                try:
                    max_power = article_soup.find('td', text='Max Power').find_next_sibling('td').get_text()
                except:
                    max_power = "Not Available"
                try:
                    
                    mileage = article_soup.find('td', text='Mileage').find_next_sibling('td').get_text()
                except:
                    mileage = "Not Available"
                result=FlipkartBikeModel(name=model_name, price=price[2:], displacement=displacement, max_power=max_power, mileage=mileage)
                result.save()
          
        messages.success(request, 'Data Saved Successfully')
    except Exception as ep:
         print(ep)
         messages.error(request, 'Data Not Saved')
    return redirect("interface_download_dataset")

def retrive_bike_flipkart(request):
      data=FlipkartBikeModel.objects.all()
      print("Printing retive data")
      response = HttpResponse('text/csv')
      response['Content-Disposition'] = 'attachment; filename=students.csv'
      writer = csv.writer(response)
      writer.writerow(['NAME', 'PRICE', 'DISPLACEMENT', 'MAX POWER', 'MILEAGE'])
     
        # writer.writerow(['NAME', 'COLOR', 'HEADPHONE TYPE', 'INLINE REMOTE', 'CONNECTIVITY', 'PRICE'])    
      studs=data.values_list('name', 'price', 'displacement', 'max_power', 'mileage')
      for std in studs:
            writer.writerow(std)
      return response
def retrive_specific_bike_flipkart(request):
        record = request.GET.get('page', 1)
        record=int(record)
        print(record)
        data=FlipkartBikeModel.objects.all()[:record-1]
        print("Printing retive data")        
        # students = Student.objects.all()
        response = HttpResponse('text/csv')
        
        response['Content-Disposition'] = 'attachment; filename=students.csv'
        writer = csv.writer(response)
        # writer.writerow(['BRAND', 'PRICE', 'MEMORY', 'DISPLAY', 'CAMERA', 'BATTERY'])
        writer.writerow(['NAME', 'PRICE', 'DISPLACEMENT', 'MAX POWER', 'MILEAGE'])
       
        # studs = data.values_list('brand', 'price', 'memory', 'display', 'camera', 'battery')
        studs=data.values_list('name', 'price', 'displacement', 'max_power', 'mileage')
        for std in studs:
            writer.writerow(std)
        return response
def delete_bike_flipkart(request):
    try:
        data= FlipkartBikeModel.objects.all().delete()
        messages.success(request, "Data Deleted Successfully")
    except:
            messages.error(request, "Data Not Deleted")
    return redirect("interface_download_dataset")


def washing_machine_flipkart(request):
    # washing_machine_data = []
    num_pages = request.GET.get('page', 1)
    num_pages=int(num_pages)
   
    print(num_pages)
    base_url = "https://www.flipkart.com"
    search_term = "washing machine"
    query_params = {"q": search_term, "otracker": "search", "otracker1": "search",
                "marketplace": "FLIPKART", "as-show": "on", "as": "off"}

# Create empty lists to store the data



# Loop through multiple pages of search results
    try:
        for page_num in range(1, num_pages): # Change 4 to the number of pages you want to scrape
        # Create the search URL for the current page
            query_params["page"] = str(page_num)
            search_url = base_url + "/search?" + "&".join([f"{key}={value}" for key, value in query_params.items()])

            # Send a GET request to the search URL and parse the HTML response
            response = requests.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all the washing machine product listings on the current page
            products = soup.find_all("div", {"class": "_2kHMtA"})
            print("product")
            print(products)

            # Loop through each product and extract the relevant data
            for product in products:
                # Extract the product URL and send a GET request to it
                product_url = base_url + product.find("a")["href"]
                response = requests.get(product_url)
                soup = BeautifulSoup(response.text, 'html.parser')

            
                try:
                    details = soup.find("table", {"class": "_14cfVK"}).find_all("li")
                    print(details)
                except:
                    pass
                try:

                # function = details[0].text.strip()
                    brand = details[1].text.strip()
                except:
                    brand="Not Available"
                try:
                
                    model_name = details[2].text.strip()
                except:
                    model_name="Not Available"
                try:

                    energy_rating = details[4].text.strip()
                except:
                    energy_rating="Not Available"
                try:
                    washing_capacity = details[5].text.strip()
                except:
                    washing_capacity="Not Available"
                try:

                    maximum_spin_speed = details[7].text.strip()
                except:
                    maximum_spin_speed="Not Available"
                try:

                    price = soup.find("div", {"class": "_30jeq3 _16Jk6d"}).text.strip()
                except:
                    price="Not Available"
                
                print(brand, model_name, energy_rating, washing_capacity, maximum_spin_speed, price)
                result=FlipkartWachingMachineModel(model=model_name, brand=brand, energy_rating=energy_rating, washing_capacity=washing_capacity, maximum_spin_speed=maximum_spin_speed, price=price[1:])
                result.save()
              
        messages.success(request, "Data Downloaded Successfully")
    except Exception as ep:
        print(ep)
        messages.error(request, "Data Not Downloaded")
    return redirect("interface_download_dataset")


def retrive_washing_machine_flipkart(request):
      data=FlipkartWachingMachineModel.objects.all()
      print("Printing retive data")
      response = HttpResponse('text/csv')
      response['Content-Disposition'] = 'attachment; filename=students.csv'
      writer = csv.writer(response)
    #   writer.writerow(['NAME', 'PRICE', 'DISPLACEMENT', 'MAX POWER', 'MILEAGE'])
      writer.writerow(['BRAND', 'MODEL', 'ENERGY RATING', 'WASHING CAPACITY', 'MAXIMUM SPIN SPEED', 'PRICE'])
      studs=data.values_list('brand', 'model', 'energy_rating', 'washing_capacity', 'maximum_spin_speed', 'price')
     
      for std in studs:
            writer.writerow(std)
      return response

def delete_washing_machine_flipkart(request):
    try:
        data= FlipkartWachingMachineModel.objects.all().delete()
        messages.success(request, "Data Deleted Successfully")
    except:
        messages.error(request, "Data Not Deleted")
      
    return redirect("interface_download_dataset")

def reterive_specific_washingmachine_flipkart(request):
     record = request.GET.get('page', 1)
     record=int(record)
     print(record)

     print("inside retirve specific record")
     data=FlipkartWachingMachineModel.objects.all()[:record-1]
     print(data)
     response = HttpResponse('text/csv')
     response['Content-Disposition'] = 'attachment; filename=students.csv'
     writer = csv.writer(response)
     writer.writerow(['BRAND', 'MODEL', 'ENERGY RATING', 'WASHING CAPACITY', 'MAXIMUM SPIN SPEED', 'PRICE'])
     studs=data.values_list('brand', 'model', 'energy_rating', 'washing_capacity', 'maximum_spin_speed', 'price')
     
     for std in studs:
            writer.writerow(std)

     return response
    #  return redirect("interface_download_dataset")