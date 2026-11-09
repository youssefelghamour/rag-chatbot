""" Prompts templates """
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate
)


""" Using the retriever to feed 2 documents as context in a prompt sent to the LLM
    to answer a user question about the subject
"""


PROMPT_CREATING_QUESTION = """
Lecture: {lecture}<br>
Title: {title}<br>
Body: {body}
"""
# Format the student's question (lecture, title, body) into a single text query
# The query is sent to the retriever, which searches ChromaDB and returns
# the 2 most relevant document chunks to provide as context to the LLM.
prompt_creating_question = PromptTemplate.from_template(
    PROMPT_CREATING_QUESTION
)



# System message
PROMPT_RETRIEVING_S = """
You are a helpful Tableau course assistant.
Answer questions based on the retrieved course content.
Do not invent information or use knowledge outside the provided context.
If the answer isn't available in the context, say so.
"""

prompt_retrieving_s = SystemMessage(
    content=PROMPT_RETRIEVING_S
)



# Human message, including the 2 retrived documents and the and formatted student question
PROMPT_TEMPLATE_RETRIEVING_H = """
Context:
{context}

Student question:
{question}
"""

prompt_template_retrieving_h = HumanMessagePromptTemplate.from_template(
    PROMPT_TEMPLATE_RETRIEVING_H
)


# The prompt sent to the LLM, including:
# System message, Human message that includes the context + user question (lecture, title, body)
chat_prompt_template_retrieving = ChatPromptTemplate.from_messages([
    prompt_retrieving_s,
    prompt_template_retrieving_h
])