from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, TokenTextSplitter
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate
)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import time
import json



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


# Load the PDF
loader_pdf = PyPDFLoader("Introduction_to_Tableau.pdf")

# Extract PDF pages into a list of LangChain Document objects
docs_list = loader_pdf.load()

# Concatenate all the pages contents into one string
string_list_concat = "\n".join(
    [doc.page_content for doc in docs_list]
)


# Split sections by heading
headers_to_split_on = [
    ("#", "Section Title"),
    ("##", "Lecture Title")
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

"""
docs_list_md_split = [
    Document(metadata={
        'Section Title': '',
        'Lecture Title': ''},
        page_content=""
    }
] 
"""
docs_list_md_split = md_splitter.split_text(string_list_concat)

# List of all contents from docs_list_md_split
string_list_split = [doc.page_content for doc in docs_list_md_split]
# print([content[:50] + "..." for content in string_list_split])



""" Trasnform the Text using a LLM so it's readable: 
    - Fix cut off words
    - Correct the words
    - Fix the punctuation
"""
if not os.path.exists("formatted_lectures.json"):
    # System prompt
    PROMPT_FORMATTING_S = """
    Improve the following Tableau lecture transcript by:
    - Splitting the text into meaningful paragraphs
    - Correcting any misplaced punctuation
    - Fixing mistranscribed words (e.g., changing 'tableaux' to 'Tableau')
    - Return plain text only.
    - Do not use Markdown formatting (no **bold**, *italics*, headings, or bullet points).
    - Do not highlight or mark corrected words.
    - Preserve the original paragraph structure where appropriate.
    - Do not add titles, introductions, explanations, or closing remarks.
    """

    # Human prompt template
    PROMPT_TEMPLATE_FORMATTING_H = """
    This is the transcript:
    {lecture_transcript}
    """


    prompt_formatting_s = SystemMessage(
        content=PROMPT_FORMATTING_S
    )

    prompt_template_formatting_h = HumanMessagePromptTemplate.from_template(
        PROMPT_TEMPLATE_FORMATTING_H
    )

    chat_prompt_template_formatting = ChatPromptTemplate.from_messages([
        prompt_formatting_s,
        prompt_template_formatting_h
    ])

    chain_formatting = (
        chat_prompt_template_formatting
        | chat
        | str_output_parser
    )

    """
    string_list_formatted = chain_formatting.batch(
        string_list_split,
        config={"max_concurrency": 3}  # 3 Active tasks/contents in parallel
    )
    """

    # Sequentially instead of parallel to avoid LLM rate limit
    string_list_formatted = []

    for lecture in string_list_split:
        result = chain_formatting.invoke({
            "lecture_transcript": lecture
        })
        string_list_formatted.append(result)
        # Add delays between batches to let LLM token count limit refresh after 6 seconds
        time.sleep(7)

    # Cache the fortmatted output from the LLM in a JSON file to avoid running the LLM request each time
    with open("formatted_lectures.json", "w", encoding="utf-8") as f:
        json.dump(string_list_formatted, f, ensure_ascii=False, indent=4)

else:
    # Load the formatted lectures from the JSON file
    with open("formatted_lectures.json", "r", encoding="utf-8") as f:
        string_list_formatted = json.load(f)

# Making sure the content matches
assert len(string_list_formatted) == len(docs_list_md_split), \
    "Mismatch between original and formatted lectures"

# Replace the original content with the new corrected content
for doc, formatted_text in zip(docs_list_md_split, string_list_formatted):
    doc.page_content = formatted_text



# Split documents into smaller chunks of 500 tokens each
token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=500,
    chunk_overlap=50
)

docs_list_tokens_split = token_splitter.split_documents(docs_list_md_split)



""" Embedding the documents, storing them in vector db, and creating a retriever to query the vector db """

embedding = FastEmbedEmbeddings()

# Create the db only if it doesn't exist
if not os.path.exists("./chroma_db"):
    vectorstore = Chroma.from_documents(
        documents=docs_list_tokens_split,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
else:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "lambda_mult": 0.7
    }
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



def format_context(dictionary):
    """ Function that formats the retrived documents into a clean text
            Input: dictionary output from the RunnableParalled
            Output: formatted text
    """
    formatted_string = ""

    retrieved_list = dictionary["context"]

    for i, document in enumerate(retrieved_list, start=1):
        formatted_string += (
            f"Document {i}\n"
            f"Section Title: {document.metadata.get('section_title', '')}\n"
            f"Lecture Title: {document.metadata.get('lecture_title', '')}\n"
            f"Content: {document.page_content}\n\n"
        )

    new_dict = dictionary.copy()
    new_dict["context"] = formatted_string

    return new_dict


chain_retrieving = (
    prompt_creating_question                     # StringPromptValue object containing the injected lecture, title and body
    | RunnableLambda(lambda x: x.to_string())    # Turn the object into a string, because the retriever needs a text query to search the vector database
    | {                                          # RunnableParallel that create the human prompt for the LLM
        "context": retriever,                    # Documents retrieved fromt chroma db by the retriever
        "question": RunnablePassthrough()        # The question including the lecture, title and body
    }
    | RunnableLambda(format_context)             # Transforms the retrived document to clean formatted text
    | chat_prompt_template_retrieving            # The prompt including system message and human message
    | chat                                       # LLM receiving the full prompt
    | str_output_parser                          # LLM's output to string
)

""" retriever receives the formatted question and returns 2 relevant documents
    RunnablePassthrough() the formatted question and returns the same formatted question
        {
            "context": [Document(page_content="", metadata={}), Document(...)],
            "question": "Lecture: ... Title: ... Body: ..."
        }
    instead of sending these documents to the LLM, we transform them into cleaner structured text:
        Document 1
        Section Title: Calculations
        Lecture Title: Custom Calculations
        Content: Tableau allows you to create calculated fields...
"""

result = chain_retrieving.invoke({
    "lecture": "Adding a custom calculation",
    "title": "Why are we using SUM here? It's unclear to me.",
    "body": "This question refers to calculating the GM%."
})

print(result)