""" Context formatting and RAG chain """
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from app.prompts import (
    prompt_creating_question,
    chat_prompt_template_retrieving
)
from app.vectorstore import retriever
from app.llm import chat, str_output_parser


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


"""
                   +-------------+                 
                   | PromptInput |                 
                   +-------------+                 
                           *                       
                  +----------------+               
                  | PromptTemplate |               
                  +----------------+               
                           *                       
                      +--------+                   
                      | Lambda |                   
                      +--------+                   
                           *                       
         +---------------------------------+       
         | Parallel<context,question>Input |       
         +---------------------------------+       
                  ***            ***               
              **                      **           
+----------------------+          +-------------+  
| VectorStoreRetriever |          | Passthrough |  
+----------------------+          +-------------+  
                  ***            ***               
                       **    **                    
         +----------------------------------+      
         | Parallel<context,question>Output |      
         +----------------------------------+      
                           *                       
                  +----------------+               
                  | format_context |               
                  +----------------+               
                           *                       
                +--------------------+             
                | ChatPromptTemplate |             
                +--------------------+             
                           *                       
                     +----------+                  
                     | ChatGroq |                  
                     +----------+                  
                           *                       
                 +-----------------+               
                 | StrOutputParser | 
                 +-----------------+                             
                           *                       
              +-----------------------+            
              | StrOutputParserOutput |            
              +-----------------------+  
"""
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