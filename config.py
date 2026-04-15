import os
from dotenv import load_dotenv
import google.generativeai as genai

# Cargar las variables del archivo .env
load_dotenv()

def configurar_ia():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    return None