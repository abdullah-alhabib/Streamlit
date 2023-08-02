# %%
# %%
#!pip install pandas
!pip install -U scikit-learn

# %%
#!pip install numpy

# %%
import pandas as pd
from sklearn.pipeline import FunctionTransformer
df= pd.read_csv("cancer.csv")
df.head()


# %%
import numpy as np
df.replace(to_replace = "Low" , value = 0 , inplace = True)
df.replace(to_replace = "Medium" , value = 1 , inplace = True)
df.replace(to_replace = "High" , value = 2 , inplace = True)
df= df.drop(columns=['index','Patient Id'])
df.head()

x=df.drop(columns=['Level'])
y= df["Level"]

#%%
# %%
def getDetails():
    from sklearn.linear_model import LinearRegression
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error,r2_score
    
    X_train,X_test,Y_train, Y_test = train_test_split(x,y, random_state=42, test_size=0.3)
    model = LinearRegression()
    model.fit(X_train,Y_train)
    print(model.coef_)
    predictionM = model.predict(X_test)
    r2=r2_score(predictionM,Y_test)
    mse= mean_squared_error(predictionM,Y_test)
    score = model.score(X_test, Y_test)
    return r2 , mse , score 

# %%
def getPrediction(predict):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error,r2_score
    
    X_train,X_test,Y_train, Y_test = train_test_split(x,y, random_state=42, test_size=0.3)
    model = LinearRegression()
    predict = np.array(predict).reshape(1,-1)
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(predict)
    model.fit(X_train,Y_train)
    print(model.coef_)
    prediction = model.predict(predict)
    predictionM = model.predict(X_test)
    r2=r2_score(predictionM,Y_test)
    mse= mean_squared_error(predictionM,Y_test)
    print(model.score(X_test, Y_test))
    return prediction 


# %%
#!pip install streamlit

# %%
import streamlit as st
st.title("Linear Regresion") 
# %%
# column_variables = [print(f"{X} ",df[X].unique()) for X in x]
# column_variables
# %%
tab1, tab2, tab3 = st.tabs(["Linear Regression", "Nan", "Nan"])

with tab1:
   
   showDf= st.checkbox("Show Details")
   ShowPM= st.checkbox("Show Performance metrics")
   if showDf: 
       st.dataframe(df)
   elif ShowPM:
       r2, mse , score = getDetails()
       col1, col2, col3 = st.columns(3)
       col1.metric("R2", np.round(r2,3))
       col2.metric("MSE", np.round(mse,3))
       col3.metric("Score", np.round(score,3))


   st.header("Linear Regresion")
   st.write("this model use linear regression to predict the level lung cancer of the patient according to patient's data")
   with st.form("my_form"):
        age=st.number_input("Your age", min_value=1)
        gender = st.radio(
    "Gender",
    ('Male', 'Female'))
        airPollution = st.slider('Air Pollution?', 0, 7, 0)
        alcoholUse=st.slider('Alcohol use?', 0, 7, 0) 
        dustAllergy= st.slider('Dust Allergy?', 0, 7, 0) 
        occuPationalHazards= st.slider('OccuPational Hazards?', 0, 7, 0) 
        geneticRisk = st.slider('Genetic Risk?', 0, 7, 0) 
        chronicLungDisease=st.slider('Chronic Lung Disease?', 0, 7, 0) 
        balancedDiet = st.slider('Balanced Diet?', 0, 7, 0)
        obesity = st.slider('Obesity?', 0, 7, 0) 
        smoking = st.slider('smoking?', 0, 7, 0)
        passiveSmoker= st.slider("Passive Smoker?", 0, 7) 
        chestPain= st.slider("Chest Pain?", 0, 7) 
        coughingofBlood= st.slider("Coughing of Blood?", 0, 7)
        Fatigue= st.slider("Fatigue?", 0, 7)
        weightLoss= st.slider("Weight Loss?", 0, 7)
        ShortnessofBreath= st.slider("Shortness of Breath?", 0, 7)
        Wheezing= st.slider("Wheezing?", 0, 7) 
        swallowingDifficulty = st.slider("Swallowing Difficulty ?", 0, 7)
        clubbingofFingerNails= st.slider("Clubbing of Finger Nails?", 0, 7)
        frequentCold= st.slider("Frequent Cold?", 0, 7)
        dryCough= st.slider("Dry Cough?", 0, 7)
        Snoring= st.slider("Snoring?", 0, 7)
        submitted = st.form_submit_button("Submit")
   # Every form must have a submit button.
   if submitted:
       if gender == "Male": 
           gender = 1
       else: gender=2
       predict= [age,gender,airPollution,alcoholUse,dustAllergy,occuPationalHazards,geneticRisk,chronicLungDisease,balancedDiet,obesity,smoking,passiveSmoker,chestPain, coughingofBlood,Fatigue,weightLoss,ShortnessofBreath,Wheezing,swallowingDifficulty,clubbingofFingerNails,frequentCold, dryCough, Snoring]
       predict = getPrediction(predict)

       if predict >= 1.5 :
           predict="High"
       elif predict <1.5 and predict>0.5:
           predict="Meduim"
       else:
           predict= "Low"
 
       st.write("The severity of cancer is",predict)
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='cancer.csv',
    mime='text/csv',
)
    

   
    



# %%


