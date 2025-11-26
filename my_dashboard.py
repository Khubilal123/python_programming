import streamlit as st
import pandas as pd
import matplotlib as plt

st.title("Streamlit app of vgu")
st.text("welcome to our dashboard")
st.write("You can write")
st.title("Streamlit app of vgu")
name = st.text_input("Enter your name : ")
age = st.number_input("Enter your age :")
st.write("Your name is ", name, "Your age is ", age)
st.selectbox("Enter your course:",["BCA","BTECH","MCA"])
if st.button("CLICK ME"):
    st.success("Clicked Button")
file = st.file_uploader("upload your file")
if file:
    content = file.read()
    st.write("File uploaded successfully!!!")

data ={"Name": ["Tom","Jack","Harry"], "Marks" : [10,20,20]}
df = pd.DataFrame(data)

st.data = pd.DataFrame({
    "marks": [10,20,20]
})
st.line_chart(data)
st.bar_chart()

subject = ["python"],["c++"],["java"]
marks = [20,10,5]

add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)




    