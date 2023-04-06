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

# **********************************************************************Admin Home Page***************************************************************
def admin_home(request):
    return render(request,"admin_home.html")
# **********************************************************************Feature Page***************************************************************
def feature(request):
    return render(request,"feature.html")
def contact_us(request):
    return render(request,"contact_us.html")
# **********************************************************************Index  Page***************************************************************
def index(request):
# **********************************************************************Index  Page***************************************************************
    return render(request,"index.html")
# **********************************************************************Logout Page***************************************************************
def logout(request):
    auth.logout(request)
    return redirect('/')
# **********************************************************************Login Page***************************************************************
def login(request):

    if request.method=='POST':
        
        u=request.POST['uname']
        p=request.POST['pswd']
        
        user=auth.authenticate(username=u,password=p)
        try:
            if user.is_staff:
                # print("Inside  ")
                auth.login(request,user)
                messages.success(request,"Login Successfull")
                return redirect('admin_home')
            
            elif user is not None:
                auth.login(request,user)
                messages.success(request,"Login Successfull")
                return redirect('user_home')
            
            else:
              
                messages.error(request,"Some error occurred")
        except:
            messages.error(request,"Invalid Login Credentials")
         
    
    return render(request, 'login.html',)

# **********************************************************************User Home Page***************************************************************
def user_home(request):
    return render(request,"user_home.html")

# *****************************************************************Feedback Page***************************************************************
def feedback(request):
    user=request.user
    data=Signup.objects.get(user=user)
    if request.method=='POST':
        firstname=request.POST['fname']

        emaiid=request.POST['femail']
        ucomment=request.POST['fcomment']
        try:
            result=Feedback(name=firstname,email=emaiid,feedback=ucomment)
            result.save()
            messages.success(request,"Feedback Submitted")
          
        except:
            messages.error(request,"Some error occurred")
           
    d={'data':data}

    return render(request,"feedback.html",d)
# **********************************************************************View Feedback Page***************************************************************
def view_feedback(request):
    data=Feedback.objects.all()
    d={'data':data}
    return render(request,'view_feedback.html',d)
# **********************************************************************Delete Feedback Page***************************************************************                         
def delete_feedback(request,id):
    data=Feedback.objects.get(id=id)
    data.delete()
    return redirect('view_feedback')
# **********************************************************************Default Regression Page***************************************************************

def default_regression(request):
    Listing=[]
    id_generator=0
    download_status=0
    error=""
    # request.session['trp'] = 40
    # Listing=pd.DataFrame()
    df = pd.DataFrame()
    if request.method == 'POST' :
        print("using post method")
        pred_col=request.POST["predcol"]
       
        model_type= request.POST.getlist('mtype')
        end=int(request.POST["size1"])
      
        u=request.FILES['datafile']
        df=pd.read_csv(u)
        print("model list")
        print("model list")
        print(model_type)
        maxcol=df.shape[1]-2
        model_name="DefaultRegression"
        for model in model_type:
            model_name=model_name+"-"+model[:5]
        print("model name is ")
        print("model name is ")
        print("model name is ",model_name)
             
        try:
            X=df.drop(columns=pred_col)
            y=df[pred_col]
            if parameter_checkup(end,maxcol,X,y):
                Listing,model=default_model_regression(X,y,1,end,model_type)

                remaining_url=(str(model_name))+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name))+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                print(remaining_url,savemodelname)
                download_status=1
            else:
               
                messages.error(request,"Max Column Exceed")

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
       
    d={'data': df.to_html(),"listing":Listing ,"download":download_status,"id_generator":id_generator}
    return render(request,"default_regression.html",d)

# **********************************************************************Default Classification Page***************************************************************
def default_classification(request):
    Listing=[]
    download_status=0
    id_generator=0
    error=""
    # Listing=pd.DataFrame()
    df = pd.DataFrame()
    if request.method == 'POST' :
        print("using post method")
        pred_col=request.POST["predcol"]
        model_type= request.POST.getlist('mtype')
        end=request.POST["size1"]
        end=int(end)
        u=request.FILES['datafile']
        df=pd.read_csv(u)
        # print(df.head())
        try:
        
            X=df.drop(columns=pred_col)
            model_name="DefaultRegression"
            for model in model_type:
                 model_name=model_name+"-"+model[:5]
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col])    
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                
                Listing,model=default_model_classifier(X,y,1,end,model_type)
                remaining_url=(str(model_name))+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name))+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                download_status=1
            else:
                   messages.error(request,"Max Column Exceed")
        except Exception as ep:
              print(ep)
              messages.error(request,ep)
        # print()
    d={'data': df.to_html(),"listing":Listing,"download":download_status,"id_generator":id_generator}
    
  
    return render(request,"default_classification.html",d)

# **********************************************************************Random FOrest Classfication Page***************************************************************
def random_forest_classification(request):
    parms={}
    Listing=[]
    download_status=0
    id_generator=0
    df = pd.DataFrame()
    ucriterion=["criterion"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
    
        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        maxdepth_value=request.POST.get("slidervalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("sample_split")
        sample_split_value=int(sample_split_value)
        n_estimator_value=request.POST.get("n_estimator")
        n_estimator_value=int(n_estimator_value)
        end=int(end)
    
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('maxfeature')
        sampling_list= request.POST.getlist('sampling')
        datafi=request.FILES['datafile']
      
        # *********************Creating Parameter************************
        ucriterion=ucriterion+criterion_list
        umaxfeature=umaxfeature+maxfeature_list
        for i in range(1,maxdepth_value,5):
                  max_depth.append(i)
        for i in range(1,n_estimator_value,5):
                    n_estimator.append(i)
        for i in range(2,sample_split_value,1):
                   sample_split.append(i)
        # **************************************************************Appending Parameter****************************************************
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
        creating_hyper_paramerter(parms,n_estimator)
        creating_hyper_paramerter(parms,sample_split)
        parms=dict(parms)
        
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        try:
        
            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=RandomForestClassifier()
                Listing,model=all_classification_model(X,y,1,end,parms,model_name,sampling_list)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
              
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
               
                download_status=1
            else:
                    
                    messages.error(request, 'Max Column Exceed')
        except Exception as ep:
              print(ep)
              messages.error(request, ep)
    d={'data': df.to_html(),"listing":Listing,"download":download_status,"id_generator":id_generator}
    return render(request,"random_forest_clasification.html",d)

# **********************************************************************Interface SVC Page***************************************************************
def interface_svc(request):
    parms={}
    Listing=[]
    download_status=0
    id_generator=0
   
    df = pd.DataFrame()
    ukernal=["kernel"]
    ugamma=["gamma"]
    udegree=["degree"]
  
    if request.method == 'POST':
    #    ******************************Getting Data From User**************************
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        degree_value=request.POST.get("degreevalue")
        degree_value=int(degree_value)
      
        kernal_list= request.POST.getlist('kernal')
        gamma_list= request.POST.getlist('gamma')
        sampling_list= request.POST.getlist('sampling')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        #
        # *********************Creating Parameter************************
        ukernal=ukernal+kernal_list
        ugamma=ugamma+gamma_list
    
        for i in range(1,degree_value,1):
                 udegree.append(i)

        # **************************************************************Appending Parameter****************************************************
        creating_hyper_paramerter(parms,ugamma)
        creating_hyper_paramerter(parms,ukernal)
        creating_hyper_paramerter(parms,udegree)
        
       
        parms=dict(parms)
        
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        #
        try:
    
            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=SVC()
            
                Listing,model=all_classification_model(X,y,1,end,parms,model_name,sampling_list)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                download_status=1
            else:
                    messages.error(request,"Max Column Exceed")
        except Exception as ep:
              print(ep)
              messages.error(request,ep)
    d={'data': df.to_html(),"listing":Listing, "download":download_status,"id_generator":id_generator}
    return render(request,"interface_svc.html",d)

# **********************************************************************Interface Gradient Boosting Page***************************************************************
def interface_gradientboosting_classifier(request):
    parms={}
    id_generator=0
    Listing=[]
    download_status=0

    df = pd.DataFrame()
    ucriterion=["criterion"]
    ulearning_rate=["learning_rate"]
    uloss=["loss"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
    # **************************Getting Data From User****************
        
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
    
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('max_features')
        loss_list= request.POST.getlist('loss')
        sampling_list= request.POST.getlist('sampling')
        datafi=request.FILES['datafile']
       
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

        # **************************************************************Appending Parameter****************************************************
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
        creating_hyper_paramerter(parms,n_estimator)
        creating_hyper_paramerter(parms,sample_split)
        creating_hyper_paramerter(parms,ulearning_rate)
        creating_hyper_paramerter(parms,uloss)
    
        parms=dict(parms)
      
 
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
    
        try:
        
            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col])    
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=GradientBoostingClassifier()
                Listing,model=all_classification_model(X,y,1,end,parms,model_name,sampling_list)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                download_status=1
            else:
                   messages.error(request,"Max Column Exceed")
        except Exception as ep:
              print(ep)
              messages.error(request,ep)
        # print()
    d={'data': df.to_html(),"listing":Listing,"download":download_status,"id_generator":id_generator}
    
     
    return render (request,"gradientboosting_classifier.html",d)
#  ****************************************************************Interface Decision Tree  Classfier Page***************************************************************
def interface_decisiontree_classifier(request):
    parms={}
    Listing=[]
    download_status=0
    id_generator=0

    df = pd.DataFrame()
    ucriterion=["criterion"]
 
    usplitter=["splitter"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
   
    sample_split=["min_samples_split"]
    if request.method == 'POST':
    # **************************Getting Data From User****************
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        
        maxdepth_value=request.POST.get("maxdepthvalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("minsamplevalue")
        sample_split_value=int(sample_split_value)
      
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('max_features')
        splitter_list= request.POST.getlist('splitter')
        sampling_list= request.POST.getlist('sampling')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        print(ucriterion,"slider value is:",)
        # *********************Creating Parameter************************
        ucriterion=ucriterion+criterion_list
        umaxfeature=umaxfeature+maxfeature_list
        usplitter=usplitter+splitter_list      
        for i in range(1,maxdepth_value,5):
                      max_depth.append(i)
      
        for i in range(1,sample_split_value,1):
                     sample_split.append(i)
        # **************************************************************Appending Parameter****************************************************
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
    
        creating_hyper_paramerter(parms,sample_split)
     
        creating_hyper_paramerter(parms,usplitter)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        try:
            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=DecisionTreeClassifier()
                Listing,model=all_classification_model(X,y,1,end,parms,model_name,sampling_list)
                
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)    
                download_status=1
            else:
                 messages.error(request,"Max Column Exceed")
        except Exception as ep:
              print(ep)
              messages.error(request,ep)
        #
    d={'data': df.to_html(),"listing":Listing ,"download":download_status,"id_generator":id_generator}
    return render(request,"decisiontree_classifier.html",d)

#  ****************************************************************Interface KNN  Classfier Page***************************************************************
def interface_knn_classifier(request):
    parms={}
    Listing=[]
    download_status=0
    id_generator=0
 
    df = pd.DataFrame()
    uweights=["weights"]
    un_neighbors=["n_neighbors"]
    ualgorithm=["algorithm"]
   
    if request.method == 'POST':
    # **************************Getting Data From User****************
        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        weights_list=request.POST.getlist("weights")
        alogorith_list= request.POST.getlist('algorithm')
        sampling_list= request.POST.getlist('sampling')
        print("The spling is ")
        print("The spling is ")
        print("The spling is ")
        print(sampling_list)
        print(type(sampling_list))
        n_neighnour_value=request.POST["kvalue"]
   
       
        print(n_neighnour_value)
        n_neighnour_value=int(n_neighnour_value)
       
        datafi=request.FILES['datafile']
        # *********************Creating Parameter************************
        uweights=uweights+weights_list
        ualgorithm=ualgorithm + alogorith_list
        
        for i in range(1,n_neighnour_value,1):
                  un_neighbors.append(i)
        # **************************************************************Appending Parameter****************************************************
        creating_hyper_paramerter(parms,uweights)
        creating_hyper_paramerter(parms,ualgorithm)
        creating_hyper_paramerter(parms,un_neighbors)
      
        
        parms=dict(parms)
        
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
                Listing,model=all_classification_model(X,y,1,end,parms,model_name,sampling_list)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
               
                download_status=1
            else:
                    messages.error(request,"Max Column Exceed")
                    
        except Exception as ep:
              print(ep)
              messages.error(request,ep)
    d={'data': df.to_html(),"listing":Listing,"download":download_status, "chosenparameter":parms,"id_generator":id_generator}
    return render(request,"knn_classifier.html",d)

# ****************************************************************Interface Random Regressor Page***************************************************************
def random_forest_regression(request):
    parms={}
    Listing=[]
    id_generator=0
    download_status=0
    df = pd.DataFrame()
    ucriterion=["criterion"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
    #    ***************Getting Parameter from User***************
   
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        maxdepth_value=request.POST.get("slidervalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("sample_split")
        sample_split_value=int(sample_split_value)
        n_estimator_value=request.POST.get("n_estimator")
        n_estimator_value=int(n_estimator_value)
      
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('maxfeature')
        datafi=request.FILES['datafile']
      
        # *********************Creating Parameter************************
        ucriterion=ucriterion+criterion_list
        umaxfeature=umaxfeature+maxfeature_list
        for i in range(1,maxdepth_value,5):
                      max_depth.append(i)
        for i in range(1,n_estimator_value,5):
                        n_estimator.append(i)
        for i in range(2,sample_split_value,1):
                     sample_split.append(i)

        # **************************************************************Appending Parameter****************************************************
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
            
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=RandomForestRegressor()
                Listing,model=all_regression(X,y,1,end,parms,model_name)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                # Listing=random_forest_regression2(X,y,1,end,parms)
                download_status=1
              
            else:
               
                messages.error(request,"Max Column Exceed")

        except Exception as ep:
           
            print(ep)
        
    d={'data': df.to_html(),"listing":Listing ,"download":download_status,"id_generator":id_generator}
    
    
    return render(request,"random_forest_regression.html",d)
# Create your views here.
# ************************************************************Interface SVM Regressor Page***************************************************************
def interface_svm(request):
    parms={}
    id_generator=0
    Listing=[]
    download_status=0
 
    df = pd.DataFrame()
    ukernal=["kernel"]
    ugamma=["gamma"]
    udegree=["degree"]
    
    if request.method == 'POST':
    #    ***************Getting Parameter from User***************
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        degree_value=request.POST.get("degreevalue")
        degree_value=int(degree_value)
      
        kernal_list= request.POST.getlist('kernal')
        gamma_list= request.POST.getlist('gamma')
        datafi=request.FILES['datafile']
        print(pred_col,end)
        
        # *********************Creating Parameter************************
        ukernal=ukernal+kernal_list
        ugamma=ugamma+gamma_list
   
        for i in range(1,degree_value,1):
                    udegree.append(i)

        # **************************************************************Appending Parameter****************************************************
        creating_hyper_paramerter(parms,ugamma)
        creating_hyper_paramerter(parms,ukernal)
        creating_hyper_paramerter(parms,udegree)        
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
     
        try:

            X=df.drop(columns=pred_col)
            
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
       
                model_name=SVR()
                Listing,model=all_regression(X,y,1,end,parms,model_name)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                download_status=1
            else:
               
                messages.error(request,"Max Column Exceed")

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
       
    d={'data': df.to_html(),"listing":Listing ,"download":download_status,"id_generator":id_generator,}
    return render(request,"interface_svm.html",d)
# *********************************************************Gradient Boosting Regressor****************************************************
def interface_gradientboosting_regressor(request):
    parms={}
    Listing=[]
    download_status=0
    id_generator=0
    
    df = pd.DataFrame()
    ucriterion=["criterion"]
    ulearning_rate=["learning_rate"]
    uloss=["loss"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
    #    ***************Getting Parameter from User***************
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
       
        criterion_list= request.POST.getlist('criteion')
        maxfeature_list= request.POST.getlist('max_features')
        loss_list= request.POST.getlist('loss')
        datafi=request.FILES['datafile']
    
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

        # **************************************************************Appending Parameter****************************************************
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
        creating_hyper_paramerter(parms,n_estimator)
        creating_hyper_paramerter(parms,sample_split)
        creating_hyper_paramerter(parms,ulearning_rate)
        creating_hyper_paramerter(parms,uloss)
        
        parms=dict(parms)
      
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
       
        try:

            X=df.drop(columns=pred_col)
            # y=le.fit_transform(df[pred_col]) 
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=GradientBoostingRegressor()
                Listing,model=all_regression(X,y,1,end,parms,model_name)
            
                          
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
            
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                download_status=1
            else:
                
                messages.error(request,"Max Column Exceed")

        except Exception as ep:
           
            messages.error(request,ep)
            print(ep)
        # print()
    d={'data': df.to_html(),"listing":Listing,"download":download_status, "id_generator":id_generator}
    
     
    return render (request,"gradientboosting_regressor.html",d)
# ********************************************************Decision Tree Regressor****************************************************
def interface_decisiontree_regressor(request):
    parms={}
    Listing=[]
    id_generator=0
    download_status=0
   
    df = pd.DataFrame()
    ucriterion=["criterion"]
 
    usplitter=["splitter"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    # n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
    #    ***************Getting Parameter from User***************
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
      
        maxdepth_value=request.POST.get("maxdepthvalue")
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST.get("minsamplevalue")
        sample_split_value=int(sample_split_value)
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
        
        for i in range(1,sample_split_value,1):
                sample_split.append(i)
       
        # **************************************************************Appending Parameter****************************************************
        creating_hyper_paramerter(parms,ucriterion)
        creating_hyper_paramerter(parms,umaxfeature)
        creating_hyper_paramerter(parms,max_depth)
      
        creating_hyper_paramerter(parms,sample_split)
        
        creating_hyper_paramerter(parms,usplitter)
        print(type(parms))
        parms=dict(parms)
        print(type(parms))
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
       
        try:

            X=df.drop(columns=pred_col)
           
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=DecisionTreeRegressor()
                Listing,model=all_regression(X,y,1,end,parms,model_name)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                Listing,model=all_regression(X,y,1,end,parms,model_name)
                download_status=1
            else:
                error="Max Column Exceed"

        except Exception as ep:
            error="Wrong Prediction Columns"
        # print()
    d={'data': df.to_html(),"listing":Listing,"download":download_status, "id_generator":id_generator}
    return render(request,"decisiontree_regressor.html",d)

# ******************************************************************KNN Regressor***********************************************
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
   
    if request.method == 'POST':
#   ******************************Getting Data from form**************************
        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        weights_list=request.POST.getlist("weights")
        alogorith_list= request.POST.getlist('algorithm')
        n_neighnour_value=request.POST["kvalue"]
   
        n_neighnour_value=int(n_neighnour_value)
        #
     
        datafi=request.FILES['datafile']
        # *********************Creating Parameter************************
        uweights=uweights+weights_list
        ualgorithm=ualgorithm + alogorith_list
        
        for i in range(1,n_neighnour_value,1):
                     un_neighbors.append(i)
       
        creating_hyper_paramerter(parms,uweights)
        creating_hyper_paramerter(parms,ualgorithm)
        creating_hyper_paramerter(parms,un_neighbors)
    
        parms=dict(parms)
        
        print("inside view",parms)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        try:

            X=df.drop(columns=pred_col)
         
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=KNeighborsRegressor()
               
                Listing,model=all_regression(X,y,1,end,parms,model_name)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                download_status=1
            else:
                   messages.error(request,"Max Column Exceed")

        except Exception as ep:
              messages.error(request,ep)

        # print()
    d={'data': df.to_html(),"listing":Listing ,"download":download_status, "chosenparameter":parms,"id_generator":id_generator}
    return render(request,"knn_regressor.html",d)
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
# ******************************************************************User Profile***********************************************
def view_user(request):
    data=Signup.objects.all()
    print(data)
    d={'data': data}
    return render(request,"view_user.html",d)
def delete_user(request,id):
    data=User.objects.get(id=id)
    data.delete()
    return redirect('view_user')
# ****************************************************************User Trained Model****************************************************
def user_trained_model(request):
     data=TrainedModel.objects.filter(username=request.user.username).order_by("-date","-time").values()
     print(f"the username is {request.user.username}")
     print("Printing Trained Model")
     print(data)
     d={'data': data}
     return render(request,"user_trained_model.html",d)
# /******************************************************* download Random Forest Regressor****************************************/
def interface_download_random_forest_regressor(request):
    download_status=0
    id_generator=0
    parameter_list=[]
    Listing=[]
    
    df = pd.DataFrame()
    ucriterion=["criterion"]
    umaxfeature=["max_features"]
    max_depth=["max_depth"]
    n_estimator=["n_estimators"]
    sample_split=["min_samples_split"]
    if request.method == 'POST':
    #   ******************************Getting Data from form**************************
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        maxdepth_value=request.POST["slidervalue"] #mnot
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST["sample_split"]
        sample_split_value=int(sample_split_value)
        n_estimator_value=request.POST["n_estimator"]
        n_estimator_value=int(n_estimator_value)
        end=int(end)
        criterion_list= request.POST['criteion']
        maxfeature_list= request.POST['maxfeature']
        datafi=request.FILES['datafile']
# *********************Creating Parameter************************
   
        parameter_list.append( end)
        parameter_list.append(criterion_list)
        parameter_list.append(maxfeature_list)
        parameter_list.append( sample_split_value)
        parameter_list.append(n_estimator_value)
        parameter_list.append(maxdepth_value)
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        
        try:

            X=df.drop(columns=pred_col)
    
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=RandomForestRegressor()
                Listing,model=download_random_forest_regressor(X,y,parameter_list)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
           
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)

                download_status=1
            
            else:
                messages.error(request,"Max Column Exceed")
             

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
       
    d={'data': df.to_html(),"download":download_status,"result":Listing,"id_generator":id_generator}
      
    return render (request,"download_random_forest_regressor.html",d)
# ******************************************************************************************************************
def interface_download_logistic_regression(request):
    download_status=0
    parameter_list=[]
    id_generator=0
    Listing=[]
    error=""
    df = pd.DataFrame()
   
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
            #  Listing=download_logistic_regresion(X,y,parameter_list)
                model_name=LogisticRegression()
                Listing,model=download_random_forest_regressor(X,y,parameter_list)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
           
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)

            #  print(Listing)
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
    d={'data': df.to_html(),"error":error ,"download":download_status,"id_generator":id_generator,"result":Listing}
    return render(request,"download_logistic_regression.html",d)

# ***********************************************************************Download SVR Regression*******************************************************************

def interface_download_svr(request):
    download_status=0
    parameter_list=[]
    id_generator=0
    Listing=[]

    df = pd.DataFrame()
 
    if request.method == 'POST':
        
        # **************************Data Reading****************
        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        ukernal=request.POST["kernal"] #mnot
       
        ugamma=request.POST["gamma"]
       
        degreevalue= int(request.POST['degreevalue'])
  
       
        datafi=request.FILES['datafile']
        # *******************Appending Parameters******************
   
        parameter_list.append( end)
        parameter_list.append(ukernal)
        parameter_list.append(ugamma)
        parameter_list.append(degreevalue)
       
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
          
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
           
                model_name=SVR()
                Listing,model=download_svr_regression(X,y,parameter_list)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
           
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)

                download_status=1
            
            else:
             messages.error(request,"Max Column Exceed")

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
        # print()
    d={'data': df.to_html() ,"download":download_status, "result":Listing,"id_generator":id_generator}

    return render(request,"download_svr_regression.html",d)

# ***********************************************************************Download KNN Regression*******************************************************************
def interface_download_knn_regression(request):
    download_status=0
    id_generator=0
    parameter_list=[]
    Listing=[]
  
    df = pd.DataFrame()
    
    if request.method == 'POST':
#  **************************Data Reading****************

        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        uweight=request.POST["weights"] 
       
        ualgorith=request.POST["algorithm"]
     
        kvalue= int(request.POST['kvalue'])
  
       
        datafi=request.FILES['datafile']
        # *******************Appending Parameters******************
   
        parameter_list.append( end)
        parameter_list.append(uweight)
        parameter_list.append(ualgorith)
        parameter_list.append(kvalue)
      
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
            
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                Listing,model=download_knn_regression(X,y,parameter_list)
                model_name=KNeighborsRegressor()
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
             
                download_status=1
            
            else:
               messages .error(request,"Max Column Exceed")

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
        #
    d={'data': df.to_html(),"download":download_status, "result":Listing,"id_generator":id_generator}
    return render(request,"download_knn_regression.html",d)

# ***********************************************************************Download Decision Tree Regression*******************************************************************
def interface_download_decsiontree_regression(request):
    download_status=0
    id_generator=0
    parameter_list=[]
  
    Listing=[]
    df = pd.DataFrame()
    
    if request.method == 'POST':
    # **************************Data Reading****************

        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        uspliter=request.POST["splitter"] #mnot
       
        ucriterion=request.POST["criterion"]
        umaxfeature=request.POST["max_features"]
       
        uminsamplevalue= int(request.POST['minsamplevalue'])
        umaxdepthvalue= int(request.POST['maxdepthvalue'])
  
       
        datafi=request.FILES['datafile']
        
        # *******************Appending Parameters******************
        parameter_list.append( end)
        parameter_list.append(uspliter)
        parameter_list.append(ucriterion)
        parameter_list.append(umaxfeature)
        parameter_list.append(uminsamplevalue)
        parameter_list.append(umaxdepthvalue)
       
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
       
        try:

            X=df.drop(columns=pred_col)
            
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                Listing,model=download_decision_regression(X,y,parameter_list)
                model_name=KNeighborsRegressor()
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                download_status=1
            
            else:
                messages.error(request,"Max Column Exceed")

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
        # print()
    d={'data': df.to_html() ,"download":download_status, "result":Listing,"id_generator":id_generator}
    return render(request,"download_decisiontree_regression.html",d)

def interface_download_gradientboosting_regressor(request):
    download_status=0
    id_generator=0
    parameter_list=[]
  
    Listing=[]
    df = pd.DataFrame()
    
    if request.method == 'POST':
        # **************************Data Reading****************
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        learing_rate_value=float(request.POST["learingvalue"])
       
        maxdepth_value=int(request.POST["maxdepthvalue"])
    
        sample_split_value=int(request.POST["minsamplevalue"])
        n_estimator_value=int(request.POST["nestimatorsvalue"])
        n_estimator_value=int(n_estimator_value)
       
        ucriterion= request.POST['criteion']
        umaxfeature= request.POST['max_features']   
        uloss= request.POST['loss']
        datafi=request.FILES['datafile']
        # *******************Appending Parameters******************
        parameter_list.append( end)
        parameter_list.append(uloss)
        parameter_list.append(ucriterion)
        parameter_list.append(umaxfeature)
        parameter_list.append(learing_rate_value)
        parameter_list.append(n_estimator_value)
        parameter_list.append(maxdepth_value)
        parameter_list.append(sample_split_value)
       
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
       
        try:

            X=df.drop(columns=pred_col)
            
            y=df[pred_col]
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                Listing,model=download_gradientboosting_regression(X,y,parameter_list)
                model_name=GradientBoostingRegressor()
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                download_status=1
            
            else:
                messages.error(request,"Max Column Exceed")

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
        # print()
    d={'data': df.to_html() ,"download":download_status, "result":Listing,"id_generator":id_generator}
    
     
    return render(request,"download_gradientboosting_regression.html",d)

# ***********************************************************************Download Decision Tree Classification*************************************************************
def interface_download_decsiontree_classfier(request):
    download_status=0
    id_generator=0
    parameter_list=[]
    Listing=[]
   
    df = pd.DataFrame()
    # sample_split=["min_samples_split"]
    if request.method == 'POST':
        # **************************Data Reading****************

        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        uspliter=request.POST["splitter"] #mnot
       
        ucriterion=request.POST["criterion"]
        umaxfeature=request.POST["max_features"]
        usampling=request.POST["sampling"]
        
        uminsamplevalue= int(request.POST['minsamplevalue'])
        umaxdepthvalue= int(request.POST['maxdepthvalue'])
  
       
        datafi=request.FILES['datafile']
        print(pred_col,end)
        # *******************Appending Parameters******************
        parameter_list.append( end)
        parameter_list.append(uspliter)
        parameter_list.append(ucriterion)
        parameter_list.append(umaxfeature)
        parameter_list.append(uminsamplevalue)
        parameter_list.append(umaxdepthvalue)
        parameter_list.append(usampling)
        
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
    
        try:

            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                Listing,model= download_decision_classfier(X,y,parameter_list)
                model_name=KNeighborsRegressor()
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
             
     

                download_status=1
            
            else:
                messages.error(request,"Max Column Exceed")

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
        # print()
    d={'data': df.to_html() ,"download":download_status, "listing":Listing,"id_generator":id_generator}
    return render(request,"download_decisiontree_classfier.html",d)
def interface_download_randomforest_classification(request):
    download_status=0
    id_generator=0
    parameter_list=[]
    Listing=[]

    df = pd.DataFrame()
   
    if request.method == 'POST':
        # **************************Data Reading****************
        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])
        maxdepth_value=request.POST["slidervalue"] #mnot
        maxdepth_value=int(maxdepth_value)
        sample_split_value=request.POST["sample_split"]
        sample_split_value=int(sample_split_value)
        n_estimator_value=request.POST["n_estimator"]
        n_estimator_value=int(n_estimator_value)
        end=int(end)
        
        criterion_list= request.POST['criteion']
        maxfeature_list= request.POST['maxfeature']
        usampling=request.POST["sampling"]
        datafi=request.FILES['datafile']
        print(pred_col,end)
        # *******************Appending Parameters******************
        parameter_list.append( end)
        parameter_list.append(criterion_list)
        parameter_list.append(maxfeature_list)
        parameter_list.append( sample_split_value)
        parameter_list.append(n_estimator_value)
        parameter_list.append(maxdepth_value)
        parameter_list.append(usampling)
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        
        try:

            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                model_name=RandomForestRegressor()
                Listing,model=download_random_forest_classification(X,y,parameter_list)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
           
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)

            
                print(Listing)
             
            #  request.session['trp'] = Listing

                download_status=1
            
            else:
                messages.error(request,"Max Column Exceed")

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
        # print()
    d={'data': df.to_html() ,"download":download_status,"listing":Listing,"id_generator":id_generator}
    return render(request,"download_random_forest_classification.html",d)

def interface_download_knn_classfier(request):
    download_status=0
    id_generator=0
    parameter_list=[]
    Listing=[]
    
    df = pd.DataFrame()
    
    if request.method == 'POST':
        # **************************Data Reading****************

        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        uweight=request.POST["weights"] #mnot
       
        ualgorith=request.POST["algorithm"]
        kvalue= int(request.POST['kvalue'])
  
        usampling=request.POST["sampling"]
        datafi=request.FILES['datafile']

        # *******************Appending Parameters******************
        parameter_list.append( end)
        parameter_list.append(uweight)
        parameter_list.append(ualgorith)
        parameter_list.append(kvalue)
        parameter_list.append(usampling)
       
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
      
        try:

            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
                Listing,model=download_knn_classfier(X,y,parameter_list)
                model_name=KNeighborsRegressor()
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)

                download_status=1
            
            else:
                messages.error(request,"Max Column Exceed")

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
        # print()
    d={'data': df.to_html(),"download":download_status, "listing":Listing,"id_generator":id_generator}
    return render(request,"download_knn_classfier.html",d)


# *************************************************************************** download Logistic Regression****************************************************

def interface_download_logistic_classfier(request):
    download_status=0
    parameter_list=[]
    id_generator=0
    Listing=[]
   
    df = pd.DataFrame()
    
    if request.method == 'POST':
        # **************************Data Reading****************)
      
       

        pred_col=request.POST["predcol"]
        end=int(request.POST["size1"])

        penalty_list=request.POST["penalty"] #mnot
       
        umulti_class_list=request.POST["multi_class"]
       
        solver_list=request.POST["solver"]
       
        
        c_value= int(request.POST['cvalue'])
        usampling=request.POST["sampling"]
       
        datafi=request.FILES['datafile']
        # *******************Appending Parameters******************
   
        parameter_list.append( end)
        parameter_list.append(penalty_list)
        parameter_list.append(umulti_class_list)
        parameter_list.append( solver_list)
        parameter_list.append(c_value)
        parameter_list.append(usampling)
        
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        # print(df.head())
        try:

            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
           
                model_name=LogisticRegression()
                Listing,model=download_logistic_classifier(X,y,parameter_list)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
           
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)

       
                download_status=1
            
            else:
                messages.error(request,"Max Column Exceed")

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
        # print()
    d={'data': df.to_html() ,"download":download_status,"id_generator":id_generator,"listing":Listing}
    return render(request,"download_logistic_classfier.html",d)
    #  *****************************************************Download SVCClassifier***************************************************
def interface_download_svc(request):
    download_status=0
    parameter_list=[]
    id_generator=0
    Listing=[]
  
    df = pd.DataFrame()
  
    if request.method == 'POST':
        # **************************Data Reading****************)
    
        pred_col=request.POST["predcol"]
        end=request.POST["size1"]
        end=int(end)
        ukernal=request.POST["kernal"] #mnot
       
        ugamma=request.POST["gamma"]
        usampling=request.POST["sampling"]
       
        # solver_list=request.POST["solver"]
       
        # ucriteion=request.POST["criteion"]
        degreevalue= int(request.POST['degreevalue'])
  
       
        datafi=request.FILES['datafile']
        print(pred_col,end)
   
        parameter_list.append( end)
        parameter_list.append(ukernal)
        parameter_list.append(ugamma)
        parameter_list.append(degreevalue)
        parameter_list.append(usampling)
      
       
        # **************************Data Reading****************
        df=pd.read_csv(datafi)
        
        try:

            X=df.drop(columns=pred_col)
            le=LabelEncoder()
            y=le.fit_transform(df[pred_col]) 
            maxcol=df.shape[1]-2
            if parameter_checkup(end,maxcol,X,y):
            #  Listing=download_svr_regression(X,y,parameter_list)
                model_name=SVR()
                Listing,model=download_svc_classfier(X,y,parameter_list)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
           
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                download_status=1
            
            else:
                messages.error(request,"Max Column Exceed")

        except Exception as ep:
            messages.error(request,ep)
            print(ep)
        # print()
    d={'data': df.to_html(),"download":download_status, "listing":Listing,"id_generator":id_generator}

    return render(request,"download_svc_classfier.html",d)


def download_model(request,id):
        print("****************************")
        print("download model id")
        print(id)
        data=TrainedModel.objects.filter(username=request.user.username,uniqueid=id).values_list('location', flat=True)
        print(f"the username is {request.user.username}")
        print("Printing Trained Model")
        print(f"The data is {data}")
        print(f"The data at 0 is {data[0]}")
        # base_url="D:/machine learning/Project/Website/Train_your_model/media/"
        remaining_url=data[0]
        print(remaining_url)
        file_path = remaining_url
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
    id_generator=0
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
        sampling_list= request.POST.getlist('sampling')
        
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
                model_name=LogisticRegression()
                Listing,model=all_classification_model(X,y,1,end,parms,model_name,sampling_list)
                remaining_url=(str(model_name))[0:-2]+str(pred_col[:3])+str(end)+str(request.user.id)+request.user.username[:3]
                savemodelname=(str(model_name)[0:-2])+"-"+str(pred_col)
                id_generator=model_saving(model,remaining_url,request.user.username,savemodelname)
                download_status=1
            else:
                    error="Max Column Exceed"
        except Exception as ep:
              print(ep)
        # print()
    d={'data': df.to_html(),"listing":Listing,"error":error ,"download":download_status, "chosendparameter":parms,"id_generator":id_generator}
    return render(request,"Logistic_regression.html",d)




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