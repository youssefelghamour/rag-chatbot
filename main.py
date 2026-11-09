from langchain_community.document_loaders.pdf import PyPDFLoader

# Load the PDF
loader_pdf = PyPDFLoader("Introduction_to_Tableau.pdf")

# Extract PDF pages into a list of LangChain Document objects
docs_list = loader_pdf.load()

# Concatenate all the pages contents into one string
string_list_concat = "\n".join(
    [doc.page_content for doc in docs_list]
)

print(string_list_concat[:2000])