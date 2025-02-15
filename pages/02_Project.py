import streamlit as st 
from PIL import Image
from footer import render_footer
#render_footer()
    #st.title("Page 2")
    #st.write("This is the page for project details")
image = Image.open('wf.png')
	    
col1, mid, col2 = st.columns([5,0.5,0.5])
with col1:
        st.image(image,use_column_width=False)
with col2:
            #showmol(xyzview,height=300,width=400)
	    st.write("")           



# Render the footer
render_footer()

st.write(""" This page will let you know the complete detail about the project... 
   


   ** For any feedback or suggestion please write me -- mushtaq.ali@kit.edu                                                                                         
   **Contact over Linkdin :** [Mushtaq Ali](https://www.linkedin.com/in/mushtaq-ali/)	""")
