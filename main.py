from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, TokenTextSplitter
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import time
import json



load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")


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


    chat = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0
    )

    str_output_parser = StrOutputParser()


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


print(len(docs_list_md_split))
print(len(docs_list_tokens_split))
