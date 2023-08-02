# %%
import pandas as pd 
import numpy as np 
df = pd.read_csv("Student_Performance.csv")
df.head()
#df.isna().sum()
df.replace(to_replace = "Yes" , value = 0 , inplace = True)
df.replace(to_replace = "No" , value = 1 , inplace = True)
# %%
import numpy as np

# Log transform the target variable.
x= np.array(df.drop(columns="Performance Index"))
y= np.array(df['Performance Index'])

#%%
# def init_wights():
#     w=np.random(x.shape[1])
#     return w

# def normlize(data):
#     mean_arr=[]
#     std_arr=[]
#     for i in range(data.shape[1]):
#         col= data[:,i]
#         s= np.std(col)
#         m= np.mean(col)
#         std_arr.append(s)
#         mean_arr.append(m)
#         for j in range(data.shape[0]):
#             data[j,i] = (data[j,i]-m)/s
#     return data,mean_arr,std_arr
# %%
def getPrediction(predict):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train, Y_test = train_test_split(x,y, random_state=42, test_size=0.3)
    model = LinearRegression()
    predict = np.array(predict).reshape(1,-1)
    model.fit(X_train,Y_train)
    coefficients = model.coef_
    prediction = model.predict(X_test)
    mse= mean_squared_error(prediction,Y_test)
    return prediction, mse
# %%
import streamlit as st 
import streamlit as st
st.title("Linear Regresion") 
st.dataframe(df)
# %%
# column_variables = [print(f"{X} ",df[X].unique()) for X in x]
# column_variables
# %%
tab1, tab2, tab3 = st.tabs(["Linear Regression", "Nan", "Nan"])

with tab1:
   st.header("Linear Regresion")
   st.write("This model use linear regression to predict the performance of the student based on several factors")
   with st.form("my_form"):
        hours=st.number_input("Hours of study", min_value=1, max_value=9)
        score = st.slider('Previous score?', 0, 100, 0)
        activity = st.radio(
            "Do you have any extracurricular activities",
            ('Yes', 'No'))
        sleepHour = st.slider('How many hours did you sleep?', 0, 9, 0)
        oldExam = st.slider('How many old exam did you solved to prepare?', 0, 10, 0)
        submitted = st.form_submit_button("Submit")
   # Every form must have a submit button.
   if submitted:
       if activity == "Yes": 
           activity = 0
       else: activity=1
       predict= [hours,score,activity,sleepHour,oldExam]
       st.write(predict)
       predict, mse = getPrediction(predict) 
       st.write(predict, mse)
    