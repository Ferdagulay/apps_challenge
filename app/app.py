import streamlit as st

st.set_page_config(page_title="Food Image Generator and Style Transfer", layout="centered")

st.title("Food Image Generation")
st.markdown("""
- Upload an image first.
- Then enter a prompt describing the new image you want to generate based on the uploaded image.
- Compare the generated results.
""")
st.markdown("In this application, you can try 3 different methods:")
st.markdown("- **Method 1**: GPT-4o-mini_and_DALL-E-3 Pipeline Base")
st.markdown("- **Method 2**: GPT-4o-mini_and_DALL-E-3 Pipeline Base (Enhanced)")
st.markdown("- **Method 3**: ")

st.info("Select a method from the menu on the left.")
