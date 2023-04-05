import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from PIL import Image
import seaborn as sns
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from  sklearn.svm import SVC,SVR

from sklearn.feature_selection import SelectKBest,chi2,RFE
from sklearn.tree import ExtraTreeClassifier,ExtraTreeRegressor,DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.over_sampling import SMOTE,ADASYN

from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import  make_pipeline,Pipeline
from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_regression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import  KNeighborsRegressor,KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import requests
from bs4 import BeautifulSoup

from imblearn.under_sampling import NearMiss,RandomUnderSampler
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
from imblearn.pipeline import Pipeline 
def data_preprocessor(X,y):
        """Function to Prepocess the data """
        numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )

        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", numeric_transformer, X.select_dtypes(np.number).columns.tolist()),
                    ("category", categorical_transformer,X.select_dtypes("object").columns.tolist()),
                ]
            )
        return preprocessor
def parameter_checkup(input_col,maxcol,x,y):

        """This is a function to check that input iternation does not exceeed availble column in filr """
        if input_col<=maxcol:
                    # st.write("insidce paramter function")
                    # model_search(x,y,3,input_col)
                    return 1
        else:
                    # st.warning(f"Number of column in file is less than{input_col}{maxcol} ")
                    return 0
def result_evaluator_regressor(model,xtest,ytest):
         ypred_model=model.predict(xtest)
         r2score=r2_score(ytest,ypred_model)
         mse=mean_absolute_error(ytest,ypred_model)
        #  return (round(r2score,2))*100,mse
         return round((r2score*100),2),round(mse,2)
def result_evaluator_classfication(model,xtest,ytest,temp,hyper=""):
         ypred=model.predict(xtest)
  
        #   ypred=model.predict(xtest)
         cmaxt=confusion_matrix(ytest,ypred,labels=[0,1])
         acuracy=accuracy_score(ytest,ypred)
         fig, ax = plt.subplots(figsize=(5,5))
         sns_heat=sns.heatmap(cmaxt,cmap="Greens",annot=True, robust=True, cbar=False, square=True, annot_kws={"size": 20}, fmt="d",ax=ax)
         plt.xlabel("Predicated")
         plt.ylabel("Actual")
         plt.title("Confusion Matrix")
        #  plt.show()
         if hyper=="no":
            result_path="media/Confusion Matrix/prajjwalp"+str(temp)
         else:
            result_path="media/Confusion Matrix/prajjwal"+str(temp)
         sns_heat.figure.savefig(result_path,)
         clrep=classification_report(ytest,ypred)
         return clrep,round((acuracy*100),2)

def dataSampling(sampling,X,y):
        preprocessor= data_preprocessor(X,y)

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
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        X_resampled=pd.DataFrame(X_resampled,columns=X.columns)
        return X_resampled,y_resampled


# import streamlit as st
# import numpy as np
# import pandas as pd
# from joblib import load
# from PIL import Image

# import streamlit as st
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import  matplotlib.pyplot as plt
# from  sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.feature_selection import SelectKBest,chi2,RFE
# from sklearn.tree import ExtraTreeClassifier,ExtraTreeRegressor
# from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
# from sklearn.model_selection import train_test_split,GridSearchCV
# from imblearn.over_sampling import SMOTE,ADASYN
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

# from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import  make_pipeline,Pipeline
# from sklearn.metrics import r2_score,mean_absolute_error
# from sklearn.compose import ColumnTransformer
# from sklearn.feature_selection import f_regression
# from sklearn.metrics import confusion_matrix,classification_report
# def data_preprocessor(X,y):
#         """Function to Prepocess the data """
#         numeric_transformer = Pipeline(
#                 steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
#             )

#         categorical_transformer = OneHotEncoder(handle_unknown="ignore")
#         preprocessor = ColumnTransformer(
#                 transformers=[
#                     ("numeric", numeric_transformer, X.select_dtypes(np.number).columns.tolist()),
#                     ("category", categorical_transformer,X.select_dtypes("object").columns.tolist()),
#                 ]
#             )
#         return preprocessor
# def result_evaluator_regressor(model,xtest,ytest):
#          ypred_model=model.predict(xtest)
#          r2score=r2_score(ytest,ypred_model)
#          mse=mean_absolute_error(ytest,ypred_model)
#          return r2score,mse
# def result_evaluator_classfication(model,xtest,ytest):
#          ypred=model.predict(xtest)
#          cmaxt=confusion_matrix(ytest,ypred)
#          clrep=classification_report(ytest,ypred)
#          return clrep,cmaxt 