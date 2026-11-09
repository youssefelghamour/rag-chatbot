""" LLM Groq model and output parser """
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

# LLM
chat = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_api_key,
    temperature=0
)

# Converts the AIMessage object AI output into a string 
str_output_parser = StrOutputParser()