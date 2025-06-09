import streamlit as st
import os
import base64
import json
from io import BytesIO
from PIL import Image
from openai import OpenAI
from datetime import datetime
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def encode_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

st.title("GPT-4.1 Mini Image Generator")

st.write("Upload an image and enter a prompt. The app will generate a new image based on your input.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
st.markdown("""
Prompt Samples
- This is a  black currant . Draw me a green pear. Use the same image drawing style in the image.
- Draw a shiny red apple illustrated in flat cartoon vector style with vibrant colors and soft lighting, no background, use the same style in the picture I provided.
- A soft red tomato with seeds, in flat cartoon vector style, no background
- Using the same illustration style as the image I uploaded, draw a vibrant green pear with a smooth surface. Match the exact visual style.
- Draw a sliced salmon fillet in the exact same illustration style as the whole salmon image I uploaded. Match the cartoon vector look, colors, and outlines.
""")


user_prompt = st.text_input("Enter your prompt")

if uploaded_file and user_prompt:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            try:
                # --- session ---
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_dir = f"output_st/output_method3/session_{timestamp}"
                os.makedirs(session_dir, exist_ok=True)

                
                uploaded_image_path = os.path.join(session_dir, f"uploaded_{timestamp}.png")
                image.save(uploaded_image_path)

                base64_img = encode_image_base64(image)

                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_prompt},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{base64_img}"}
                            ]
                        }
                    ],
                    tools=[{"type": "image_generation"}],
                )

                outputs = [output.result for output in response.output if output.type == "image_generation_call"]

                if outputs:
                    generated_image_path = os.path.join(session_dir, f"generated_{timestamp}.png")
                    with open(generated_image_path, "wb") as f:
                        f.write(base64.b64decode(outputs[0]))

                    # JSON metadata 
                    parsed = {
                        "user_prompt": user_prompt,
                        "generated_image_path": generated_image_path,
                        "image_path": uploaded_image_path
                    }
                    json_path = os.path.join(session_dir, f"data_{timestamp}.json")
                    with open(json_path, "w") as json_file:
                        json.dump(parsed, json_file, indent=4)

                    st.image(generated_image_path, caption="Generated Image", use_column_width=True)
                    st.success(f"Image saved to {generated_image_path}")
                    st.info(f"Uploaded image saved to {uploaded_image_path}")
                    st.info(f"Metadata JSON saved to {json_path}")

                else:
                    st.error("No generated image output found.")

            except Exception as e:
                st.error(f"Error: {e}")
