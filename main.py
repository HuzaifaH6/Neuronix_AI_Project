import os
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, set_default_openai_client, set_tracing_disabled, set_default_openai_api, ModelSettings

from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError ("Gemini API key not set in .env folder")

set_tracing_disabled(True)
set_default_openai_api("chat_completions")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(external_client)

agent = Agent ( name="Greet Agent",
                instructions="""
                You are a warm, friendly, and positive greeting agent.
                Your main goal is to greet users politely and cheerfully and also answer some general questions.
                If the user says something not related to greetings, reply with:
                "I'm a simple chatbot here to greet you! 😊"
                Keep replies short, positive, and natural.
                Detect the users language automatically.Always reply in the same language the user used (for example, reply in Roman-Urdu if the user greets in Roman-Urdu).
"""
,
                model="gemini-2.0-flash",
                model_settings=ModelSettings(
                temperature=2,
                max_tokens=300)
)

@cl.on_chat_start
async def start():
    cl.user_session.set("history",[])
    await cl.Message(content="Hello, how can I help you today?").send()

@cl.on_message
async def message_handler(message: cl.Message):

    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    cl.user_session.set("history", history)

    result = await Runner.run(
        agent,
        input=history,
    )
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()
