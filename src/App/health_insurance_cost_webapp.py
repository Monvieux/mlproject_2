import os
import streamlit as st
import joblib

# Get path of files
pfolder = os.getcwd()
fmodel = os.path.join(pfolder,'src','Models','health_insurance_cost')

def main():
    html_temp="""
        <div style="background-color:lightblue;padding:16px">
        <h2 stylr="color:black"; text-align:center>Health Insurance Cost Prediction Using ML</h2>
        </div>
"""
    st.markdown(html_temp,unsafe_allow_html=True)
    p1 = st.slider('Enter Your Age',18,100)
    s1 = st.selectbox('Sex',('Male','Female'))

    if s1=='Male': p2=1
    else: p2=0

    p3 = st.number_input('Enter your BMI :',20,50)

    p4 = st.slider('Number of children',0,10)

    s2 = st.selectbox('Are you smoker ? :',('Yes','No'))

    if s2=='Yes': p5=1
    else: p5=0

    s3 = st.selectbox('Region :',('southwest', 'southeast', 'northwest', 'northeast'))
    if s3=='southwest': p6=1
    if s3=='southeast': p6=2
    if s3=='northwest': p6=3
    if s3=='northeast': p6=4

    if st.button('Predict'):
        model = joblib.load(fmodel)
        pred = model.predict([[p1,p2,p3,p4,p5,p6]])
        st.success("Your Insurance Cost is {} $".format(round(pred[0],2)))


if __name__=='__main__':
    main()