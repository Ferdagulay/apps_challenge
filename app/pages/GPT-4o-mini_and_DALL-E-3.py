import streamlit as st
import os
import base64
import requests
import json
import re
import time
from io import BytesIO
from PIL import Image
from openai import OpenAI
from datetime import datetime
from config import OPENAI_API_KEY



client = OpenAI(api_key=OPENAI_API_KEY)
image_extensions = (".jpg", ".jpeg", ".png")

# Prompt template
prompt_template = """
You are a image captioning assistant helping to generate training captions for a dataset of foods images.
You will receive a food image and a user prompt describing how it should be redesigned.

Task:
Generate a **detailed caption** describing the image. The caption should:
- Include visual details like color, pattern, drawing style, background if visible.
- Analyze the "category"(the general type of food depicted) and describe its key features.
- Be a single sentence, fluent, and descriptive.
- These captions will be used as input for the DALL·E model. Generate captions suitable as prompts for the DALL·E model.

Use this format:
{{
  "category": "<your generated food category here>",
  "caption": "<your generated caption here>",
  "image_path": "<actual image path>",
  "user_prompt": "<user prompt>",
  "generated_image_path":"<your generated image path>"
}}
"""

st.title("GPT-4o-mini + DALL·E-3 Image Generator Base")
st.write("Upload a food image and provide a redesign prompt. A new image will be created based on your prompt and your uploaded image.")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
st.markdown("""
Prompt Samples
- This is a  black currant . Draw me a green pear. Use the same image drawing style in the image.
- Draw a shiny red apple illustrated in flat cartoon vector style with vibrant colors and soft lighting, no background, use the same style in the picture I provided.
- A soft red tomato with seeds, in flat cartoon vector style, no background
- Using the same illustration style as the image I uploaded, draw a vibrant green pear with a smooth surface. Match the exact visual style.
- Draw a sliced salmon fillet in the exact same illustration style as the whole salmon image I uploaded. Match the cartoon vector look, colors, and outlines.
""")
user_prompt = st.text_input("Enter your redesign prompt")

if uploaded_file and user_prompt:
    # Create timestamp and directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_st/outputs/session_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded image
    image = Image.open(uploaded_file)
    uploaded_image_path = os.path.join(output_dir, f"uploaded_{timestamp}.png")
    image.save(uploaded_image_path)
    st.image(image, caption="Uploaded Image", width=512)

    # Encode image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    formatted_prompt = prompt_template.format(user_prompt=user_prompt)

    with st.spinner("Generating caption with GPT-4o-mini..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": formatted_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0
            )

            raw_output = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            parsed = json.loads(json_match.group(0))

            st.success("Caption generated!")
            st.write("**Category**:", parsed["category"])
            st.write("**Caption**:", parsed["caption"])

            full_prompt = f"{user_prompt}, render in the exact same illustration style and details as described: {parsed['caption']}"
            st.info(f"Prompt for DALL·E: {full_prompt}")

            with st.spinner("Generating image with DALL·E 3..."):
                dalle_response = client.images.generate(
                    model="dall-e-3",
                    prompt=full_prompt,
                    n=1,
                    size="1024x1024"
                )

                time.sleep(5)  # to ensure URL becomes available
                generated_image_url = dalle_response.data[0].url
                st.image(generated_image_url, caption="Generated Image", width=512)

                # Save generated image
                generated_image_path = os.path.join(output_dir, f"generated_{timestamp}.png")
                img_response = requests.get(generated_image_url)
                if img_response.status_code == 200:
                    with open(generated_image_path, "wb") as f:
                        f.write(img_response.content)
                else:
                    st.warning("Failed to download the generated image.")

                # Update JSON
                parsed["user_prompt"] = user_prompt
                parsed["generated_image_path"] = generated_image_path
                parsed["image_path"] = uploaded_image_path

                # Save JSON
                json_path = os.path.join(output_dir, f"{timestamp}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)

                st.success("All outputs saved.")
                st.json(parsed)

        except Exception as e:
            st.error(f"Error: {str(e)}")
