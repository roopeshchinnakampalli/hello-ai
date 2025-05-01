import google.generativeai as genai
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import os
from dotenv import load_dotenv
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("Error: Please set GOOGLE_API_KEY in your environment variables")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def get_voice_input():
    """Capture voice input and convert it to text"""
    recognizer = sr.Recognizer()
    fs = 44100  # sample rate
    
    try:
        # Record audio in chunks
        audio_data = sd.rec(int(2 * fs), samplerate=fs, channels=1)
        sd.wait()
        
        # Convert to audio data
        audio_data = (audio_data * 32767).astype(np.int16)
        audio_data = audio_data.tobytes()
        audio = sr.AudioData(audio_data, fs, 2)
        
        try:
            text = recognizer.recognize_google(audio)
            if text:
                print(f"\nYou said: {text}")
                return text
        except sr.UnknownValueError:
            pass  # Ignore when no speech is detected
        except sr.RequestError as e:
            print(f"Error in speech recognition: {e}")
            
    except Exception as e:
        print(f"Error recording audio: {e}")
    return None

def get_gemini_response(user_input, context=None):
    """Get response from Gemini model"""
    try:
        # If context is provided, prepend it to the user input
        if context:
            user_input = f"Context: {context}\n\nUser: {user_input}"
        
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        print(f"Error getting response from Gemini: {e}")
        return None

def main():
    """Main chat loop"""
    print("\n=== Gemini Voice Chat ===")
    print("Would you like to set a context for this conversation?")
    print("1. Yes")
    print("2. No")
    
    choice = input("Enter your choice (1 or 2): ")
    context = None
    
    if choice == "1":
        print("\nPlease speak your context (what this conversation will be about):")
        context = get_voice_input()
        if context:
            print(f"\nContext set: {context}")
        else:
            print("\nCould not understand context. Starting without context.")
    
    print("\nListening continuously... Say 'quit' to exit")
    
    while True:
        # Get voice input
        user_input = get_voice_input()
        
        if user_input:
            if user_input.lower() in ['quit', 'exit', 'goodbye', 'bye']:
                print("\nGoodbye! Have a great day!")
                break
                
            # Get response from Gemini
            response = get_gemini_response(user_input, context)
            if response:
                print(f"\nGemini: {response}")
                # Use system's say command for text-to-speech
                os.system(f'say "{response}"')

if __name__ == "__main__":
    main() 
