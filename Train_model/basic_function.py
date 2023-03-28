import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from PIL import Image

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
         return (round(r2score,2))*100,mse
def result_evaluator_classfication(model,xtest,ytest):
         ypred=model.predict(xtest)
         acuracy=accuracy_score(ytest,ypred)
         clrep=classification_report(ytest,ypred)
         return clrep,acuracy




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