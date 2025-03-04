import gradio as gr
import openai
import os
from dotenv import load_dotenv

load_dotenv()
# Configuration de l'API Azure OpenAI

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

def chatbot(user_input, history=[]):
    """Fonction pour interagir avec Azure OpenAI."""
    messages = [{"role": "system", "content": "Vous êtes un assistant AI utile."}]
    
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    messages.append({"role": "user", "content": user_input})
    
    response = openai.ChatCompletion.create(
        engine=os.getenv("DEPLOYMENT_NAME"),
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        stop=None
    )
    
    bot_reply = response["choices"][0]["message"]["content"].strip()
    return bot_reply

# Interface Gradio
demo = gr.ChatInterface(fn=chatbot, title="Chat with Elpidio")

demo.launch(share=True)