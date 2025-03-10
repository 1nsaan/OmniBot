import telebot
import os
import requests
import json
import google.generativeai as genai
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load API keys from environment variables
TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 4000,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

convo = model.start_chat(history=[])

# Initialize Telegram bot
bot = telebot.TeleBot(TELEGRAM_API_KEY)

# Intent detection functions
def is_image_request(user_input):
    image_keywords = ["image", "img", "picture", "photo", "draw", "create", "generate"]
    return any(keyword in user_input.lower() for keyword in image_keywords)

def is_weather_request(user_input):
    weather_keywords = ["weather", "temperature", "forecast"]
    return any(keyword in user_input.lower() for keyword in weather_keywords)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_input = message.text.lower()
        print(f"Received user input: {user_input}")

        if is_image_request(user_input):
            prompt = user_input
            image_keywords = ["image", "img", "picture", "photo", "draw", "create", "generate"]
            for keyword in image_keywords:
                prompt = prompt.replace(keyword, "").strip()

            if not prompt:
                prompt = "a beautiful landscape"

            print(f"Image prompt: {prompt}")
            bot.reply_to(message, "Generating image...")

            model_id = "dreamlike-art/dreamlike-photoreal-2.0"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            image = pipe(prompt).images[0]
            bot.send_photo(message.chat.id, image)

        elif is_weather_request(user_input):
            match = re.search(r"weather\s*(?:in\s*)?([a-zA-Z\s]+)", user_input)
            if match:
                city_name = match.group(1).strip()
                print(f"Weather city: {city_name}")
                response = requests.get(
                    f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={OPENWEATHER_API_KEY}"
                )
                data = json.loads(response.text)
                if data["cod"] == 200:
                    temp = data["main"]["temp"] - 273.15
                    weather_response = f"Temperature in {city_name}: {temp:.1f}°C"
                    bot.reply_to(message, weather_response)
                else:
                    bot.reply_to(message, "City not found.")
            else:
                bot.reply_to(message, "Please specify a city.")

        else:
            prompt = f"{message.text}\n\nKeep your response short and concise (under 500 characters if possible)."
            convo.send_message(prompt)
            response = convo.last.text
            print(f"Gemini response: {response}")
            if response is None:
                bot.reply_to(message, "Sorry, I couldn't generate a response.")
            else:
                bot.reply_to(message, response)

    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request.")

if __name__ == "__main__":
    print("Bot is starting...")
    bot.infinity_polling()