import streamlit as st
import numpy as np
from PIL import Image
import time
from scipy.spatial.distance import cdist

st.title('Similar Images Finder') #Streamlit Title


@st.cache_data
def read_data():
    all_vecs = np.load("all_vecs.npy")  #TO Load Vectors Model
    all_names = np.load("all_names.npy") #TO Load Names Model
    return all_vecs , all_names # To Return the files imported

vecs , names = read_data() #Reading the data in ML model

_ , fcol2 , _ = st.columns(3) #Fcol2 represents Column width i.e. 3rd grid

scol1 , scol2 = st.columns(2) #Fcol2 represents Column width i.e. 2nd grid

ch = scol1.button("Start / change") #Buttons
fs = scol2.button("find similar")

if ch:
    random_name = names[np.random.randint(len(names))]
    fcol2.image(Image.open("DATASET/DATASET/images/" + random_name))
    st.session_state["disp_img"] = random_name
    st.write(st.session_state["disp_img"])
if fs:
    c1 , c2 , c3 , c4 , c5 = st.columns(5)
    idx = int(np.argwhere(names == st.session_state["disp_img"]))
    target_vec = vecs[idx]
    fcol2.image(Image.open("DATASET/DATASET/images/" + st.session_state["disp_img"]))
    top5 = cdist(target_vec[None , ...] , vecs).squeeze().argsort()[2:8]
    c1.image(Image.open("DATASET/DATASET/images/" + names[top5[0]]))
    c2.image(Image.open("DATASET/DATASET/images/" + names[top5[1]]))
    c3.image(Image.open("DATASET/DATASET/images/" + names[top5[2]]))
    c4.image(Image.open("DATASET/DATASET/images/" + names[top5[3]]))
    c5.image(Image.open("DATASET/DATASET/images/" + names[top5[4]]))

# for finding similar image as Uplaoded
#UNDER DEVELOPMENT
# if ch:
#     # random_name = names[np.random.randint(len(names))]
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         try:
#             # Display the uploaded image
#             fcol2.image(Image.open(uploaded_file))
#             st.image("disp_img", caption='Uploaded Image', use_column_width=True)
#             st.session_state["disp_img"] = uploaded_file
#             st.write(st.session_state["disp_img"])
#         except Exception as e:
#             st.write("Error:", e)

# if fs:
#     c1 , c2 , c3 , c4 , c5 = st.columns(5)
#     idx = int(np.argwhere(names == st.session_state["disp_img"]))
#     target_vec = vecs[idx]
#     fcol2.image(Image.open("./DATASET/images 2/" + st.session_state["disp_img"]))
#     top5 = cdist(target_vec[None , ...] , vecs).squeeze().argsort()[1:6]
#     c1.image(Image.open("./DATASET/images 2/" + names[top5[0]]))
#     c2.image(Image.open("./DATASET/images 2/" + names[top5[1]]))
#     c3.image(Image.open("./DATASET/images 2/" + names[top5[2]]))
#     c4.image(Image.open("./DATASET/images 2/" + names[top5[3]]))
#     c5.image(Image.open("./DATASET/images 2/" + names[top5[4]]))







    