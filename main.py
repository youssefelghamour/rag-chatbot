from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter


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

docs_list_md_split = md_splitter.split_text(string_list_concat)


for doc in docs_list_md_split[:3]:
    print("Metadata:", doc.metadata)
    print("Content:", doc.page_content[:300])
    print("-" * 50)