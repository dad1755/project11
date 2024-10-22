import streamlit as st
from PIL import Image
import pytesseract  # Importing pytesseract for OCR
#import base64
from io import BytesIO
import openai
import tiktoken  # Importing the tiktoken library

# Set up OpenAI API key from st.secrets
openai.api_key = st.secrets["openai"]["api_key"]
# Function to get response from OpenAI based on the extracted text
def get_text_response(extracted_text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                #{"role": "user", "content": "Provide answer. There are objective, and subjective. For subjective simple word conversational"},
                {"role": "user", "content": "what is the text in the picture"},
                #{"role": "user", "content": "Please provide the answer. If subjective, provide simple conversational answer."},
                {
                    "role": "user",
                    "content": extracted_text,
                },
            ],
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error getting response from OpenAI: {e}"

# Function to calculate the token count accurately
def calculate_token_count(messages):
    enc = tiktoken.encoding_for_model("gpt-4o-mini")  # Use the appropriate model
    token_count = 0
    for message in messages:
        token_count += len(enc.encode(message['content']))  # Accurate token count
    return token_count

# Streamlit App Layout
st.set_page_config(page_title="Dad24", layout="centered")

# Title and Description
st.title("exam paper extractor")
st.write("Upload an image to analyze.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Resize the image (you can adjust the size as needed)
    max_size = (900, 900)  # Set maximum width and height to reduce data size
    image.thumbnail(max_size)

   # Display the image
    st.image(image, caption='Uploaded Image', width=600)  # Width can be adjusted

    # Use OCR to extract text from the image
    extracted_text = pytesseract.image_to_string(image)

    st.write(extracted_text)  # Display the extracted text

    # Prepare messages for token count
    messages = [
        {"role": "user", "content": "Answer any questions in the following text."},
        {"role": "user", "content": extracted_text}
    ]

    # Calculate token count
    total_tokens = calculate_token_count(messages)

    # Display token count
    st.write(f"Total tokens for this request: {total_tokens}")

    # Check if token count exceeds limits and handle accordingly
    if total_tokens > 128000:
        st.error("The total token count exceeds the model's maximum context length. Please reduce the length of the messages.")
    else:
        # Get response from OpenAI when the image is uploaded
        with st.spinner("Analyzing image..."):
            extracted_data = get_text_response(extracted_text)

        # Display the extracted data
        st.subheader("Extracted Details:")
        st.write(extracted_data)
