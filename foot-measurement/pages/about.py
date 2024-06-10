import streamlit as st


def show():
    st.title("How to Use? ")
    st.write(
        """
    This application measures your foot size and suggests shoe sizes based on the image you upload.

    Here are the steps to use this application:

    1. photo of the foot in a position to step on 
    or above A4 paper with the heel at the end of the paper 
    2. photo taken with a distance of 30 cm.
    3. photos are supported by good lighting so that the feet can be seen clearly
    4. photo formats are png/jpg/jpeg
    5. Check the results and follow the suggested shoe size for a better fit.

    Enjoy using the application!
    """
    )
