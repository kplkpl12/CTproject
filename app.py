import streamlit as st
import streamlit.components.v1 as htmlviewr

with open("./ct1_2.html", "r", encoding="utf-8") as f:
    html1 = f.read()
    f.close()

with open("./ct2_5.html", "r", encoding="utf-8") as f:
    html2 = f.read()
    f.close()

with open("./ct3_1.html", "r", encoding="utf-8") as f:
    html3 = f.read()
    f.close()

# Title Msg##1
st.title(("this is Juhee Webapp!! "))
col1, col2 = st.columns((4, 1))
with col1:
    with st.expander("CT Problem1"):
        htmlviewr.html(html1, height=800)

    with st.expander("CT Problem2"):
        htmlviewr.html(html2, height=800)

    with st.expander("CT Problem3"):
        htmlviewr.html(html3, height=1200)


with col2:
    with st.expander("Tips"):
        st.info("Tips..")

st.markdown("<hr>", unsafe_allow_html=True)
st.write(
    '<font color="BLUE">(c)copyright. all rights reserved by skykang',
    unsafe_allow_html=True,
)
