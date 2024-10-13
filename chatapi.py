#Owner - Kamal
import os
import google.generativeai as genai

api_key = "AIzaSyAXd_GrVgQj0TVFp5qX_D8_PdbZBPm8wlI" 
genai.configure(api_key=api_key)


generation_config = {
    "temperature": 0.7,  
    "top_p": 0.9,
    "max_output_tokens": 200,  
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",  
    generation_config=generation_config,
)

chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "hi\n",
            ],
        },
        {
            "role": "model",
            "parts": [
                "Welcome! Let's help you find the perfect outfit always in 2 lines (indian english). Please answer the following questions:\n1. What is your gender? (Male, Female, Non-binary, Prefer not to say)\n2. Which type of clothing are you looking for? (e.g., casual, formal, business, sporty, etc.)\n3. Do you have any preferences for colors or styles? Let us know your favorite colors or styles.\n\nBased on your responses, we will suggest an outfit tailored to your preferences!",
            ],
        },
    ]
)

while True:
    user_prompt = input("User: ")
    if user_prompt.lower() == "quit":
        break  
    response = chat_session.send_message(user_prompt)
    print()
    print(response.text)