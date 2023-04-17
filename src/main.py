import asyncio
from EdgeGPT import Chatbot, ConversationStyle
import json
import re
import whisper
import speech_recognition as sr
import boto3
import pydub
from pydub import playback
import openai

# Initialize OpenAI API
openai.api_key="INSERT_YOUR_OPENAI_API_KEY_HERE"

# Create a recognizer object and wake word variable
recognizer = sr.Recognizer()
BING_WAKE_WORD= "mario"
GPT_WAKE_WORD= "luigi"

def get_wake_word(phrase):
    if BING_WAKE_WORD in phrase.lower():
        return BING_WAKE_WORD
    elif GPT_WAKE_WORD in phrase.lower():
        return GPT_WAKE_WORD
    else:
        return None

def synthesize_speech(text, output_filename):
    # session= boto3.Session(profile_name= "default")
    polly= boto3.client("polly", region_name= "eu-central-1")
    response= polly.synthesize_speech(
        Text= text,
        TextType= "text",
        OutputFormat= "mp3",
        VoiceId= "Bianca",
        Engine= "standard", # standard or neural
        LanguageCode= "it-IT",
    )

    with open(output_filename, "wb") as f:
        f.write(response["AudioStream"].read())

def play_audio(filename):
    sound= pydub.AudioSegment.from_file(filename, format= "mp3")
    playback.play(sound)

async def main():
    while True:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print(f"In attesa della wake word 'attiva mario' o 'attiva luigi'...")
            while True:
                audio= recognizer.listen(source)
                try:
                    with open("wake_audio.wav", "wb") as f:
                        f.write(audio.get_wav_data())
                    # Use the preloaded tiny_model
                    model= whisper.load_model("tiny")
                    result= model.transcribe("wake_audio.wav", fp16= False, language= "it")
                    phrase= result["text"]
                    print(f"You: said: {phrase}")

                    wake_word= get_wake_word(phrase)
                    if wake_word is not None:
                        break
                    else:
                        print("Non è una wake word. Prova ancora.")
                except Exception as e:
                    print(f"Error transcribing wake audio: {0}".format(e))
                    continue

            print("Speak a prompt...")
            synthesize_speech("Come posso aiutarti?", "wake_response.mp3")
            play_audio("wake_response.mp3")
            audio= recognizer.listen(source)

            try:
                with open("prompt_audio.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                model= whisper.load_model("base")
                result= model.transcribe("prompt_audio.wav", fp16= False, language= "it")
                user_input= result["text"]
                print(f"Hai detto: {user_input}")
            except Exception as e:
                print(f"Error transcribing prompt audio: {0}".format(e))
                continue

            if wake_word == BING_WAKE_WORD:
                with open('./cookies.json', 'r') as f:
                    cookies = json.load(f)
                bot= Chatbot(cookies=cookies)
                response= await bot.ask(prompt= user_input, conversation_style= ConversationStyle.precise)

                for message in response["item"]["messages"]:
                    if message["author"] == "bot":
                        bot_response = message["text"]

                bot_response = re.sub('\[\^\d+\^\]', '', bot_response)
                # Select only the bot response from the response dictionary
                for message in response["item"]["messages"]:
                    if message["author"] == "bot":
                        bot_response = message["text"]
                # Remove [^#^] citations in response
                bot_response = re.sub('\[\^\d+\^\]', '', bot_response)

            else:
                # Send prompt to GPT-3.5-turbo API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content":
                            "You are a helpful assistant."},
                        {"role": "user", "content": user_input},
                    ],
                    temperature=0.5,
                    max_tokens=1000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=1,
                    stop=["\nUser:"],
                )

                bot_response = response["choices"][0]["message"]["content"]

        print(f"Bot's response: {bot_response}")
        synthesize_speech(bot_response, "prompt_response.mp3")
        play_audio("prompt_response.mp3")
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main())

