import streamlit as st
from model import TextClassificationModel
from clf1 import predict

# streamlit run app.py
st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("sentiment prediction demo")
st.write("")

text = st.text_input("please input text",value = None)

if text is not None:

    st.write("")
    st.write("Just a second...")
    labels = predict(text)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
