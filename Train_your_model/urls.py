"""Train_your_model URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
# from Train_model import views.*
from  Train_model.views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path("",index,name="index"),
    path("login",login,name="login"),
    path("logout",logout,name="logout"),
    path("admin_home",admin_home,name="admin_home"),
    path("user_home",user_home,name="user_home"),
    path("signup",signup,name="signup"),
    path("user_trained_model",user_trained_model,name="user_trained_model"),
    path("default_classification",default_classification,name="default_classification"),
    path("default_regression",default_regression,name="default_regression"),
    path("random_forest_classification",random_forest_classification,name="random_forest_classification"),
    path("Extratreeclassification",Extratreeclassification,name="Extratreeclassification"),
    path("interface_svc",interface_svc,name="interface_svc"),
    path("interface_gradientboosting_classifier",interface_gradientboosting_classifier,name="interface_gradientboosting_classifier"),
    path("interface_decisiontree_classifier",interface_decisiontree_classifier,name="interface_decisiontree_classifier"),
    path("interface_knn_classifier",interface_knn_classifier,name="interface_knn_classifier"),

    path("interface_gradientboosting_regressor",interface_gradientboosting_regressor,name="interface_gradientboosting_regressor"),
    path("random_forest_regression",random_forest_regression,name="random_forest_regression"),
    path("extratreeregression",extratreeregression,name="extratreeregression"),
    path("interface_logistic_regression",interface_logistic_regression,name="interface_logistic_regression"),
    path("interface_Knn_regresso",interface_Knn_regressor,name="interface_Knn_regresso"),
    path("interface_svm",interface_svm,name="interface_svm"),
    path("interface_decisiontree_regressor",interface_decisiontree_regressor,name="interface_decisiontree_regressor"),
    path("download_random_forest_regressor",interface_download_random_forest_regressor,name="download_random_forest_regressor"),
    path("interface_download_logistic_regression",interface_download_logistic_regression,name="interface_download_logistic_regression"),
    path("interface_download_extreetree_regression",interface_download_extreetree_regression,name="interface_download_extreetree_regression"),
    path("interface_download_knn_regression",interface_download_knn_regression,name="interface_download_knn_regression"),
    path("interface_download_svr",interface_download_svr,name="interface_download_svr"),
    # path("page_reg",page_reg,name="page_reg"),
    # rough
    path("rough_regression",rough_regression,name="rough_regression"),
    path("download_model/<int:id>",download_model,name="download_model"),
    path("delete_model/<int:id>",delete_model,name="delete_model"),


    path("interface_download_dataset",interface_download_dataset,name="interface_download_dataset"),
    path("user_download_dataset",user_download_dataset,name="user_download_dataset"),


    path("mobilephone_flipkart",mobilephone_flipkart,name="mobilephone_flipkart"),
    path("retrivemobilephone_flipkart",retrivemobilephone_flipkart,name="retrivemobilephone_flipkart"),
    path("retrive_specific_mobilephone_flipkart",retrive_specific_mobilephone_flipkart,name="retrive_specific_mobilephone_flipkart"),
    path("delete_mobilephone_flipkart",delete_mobilephone_flipkart,name="delete_mobilephone_flipkart"),


    path("laptop_flipkart",laptop_flipkart,name="laptop_flipkart"),
    path("retrivelaptop_flipkart",retrivelaptop_flipkart,name="retrivelaptop_flipkart"),
    path("retrive_specific_laptop_flipkart",retrive_specific_laptop_flipkart,name="retrive_specific_laptop_flipkart"),
    

    path("delete_laptop_flipkart",delete_laptop_flipkart,name="delete_laptop_flipkart"),
     

    

    path("interface_television_flipkart",interface_television_flipkart,name="interface_television_flipkart"),
    path("retrive_television_flipkart",retrive_television_flipkart,name="retrive_television_flipkart"),
    path("retrive_specific_telivision_flipkart",retrive_specific_telivision_flipkart,name="retrive_specific_telivision_flipkart"),
    path("delete_television_flipkart",delete_television_flipkart,name="delete_television_flipkart"),   


    # *******************************************************Earphone**************************************************************
    path("earphone_flipkart",earphone_flipkart,name="earphone_flipkart"),
    path("retrive_earphone_flipkart",retrive_earphone_flipkart,name="retrive_earphone_flipkart"),
    path("retrive_specific_earphone_flipkart",retrive_specific_earphone_flipkart,name="retrive_specific_earphone_flipkart"),
    path("delete_earphone_flipkart",delete_earphone_flipkart,name="delete_earphone_flipkart"),


# **************************************************Bike****************************************************************
    path("bike_flipkart",bike_flipkart,name="bike_flipkart"),         
    path("retrive_bike_flipkart",retrive_bike_flipkart,name="retrive_bike_flipkart"),     
    path("retrive_specific_bike_flipkart",retrive_specific_bike_flipkart,name="retrive_specific_bike_flipkart"),    
    path("delete_bike_flipkart",delete_bike_flipkart,name="delete_bike_flipkart"),


# **************************************************Washing Machine****************************************************************
    path("washing_machine_flipkart",washing_machine_flipkart,name="washing_machine_flipkart"),
    path("retrive_washing_machine_flipkart",retrive_washing_machine_flipkart,name="retrive_washing_machine_flipkart"),
    path("delete_washing_machine_flipkart",delete_washing_machine_flipkart,name="delete_washing_machine_flipkart"),
    path("reterive_specific_washingmachine_flipkart",reterive_specific_washingmachine_flipkart,name="reterive_specific_washingmachine_flipkart"),






]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
