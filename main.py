from langchain_community.document_loaders import PyPDFLoader

file_path = "Think-And-Grow-Rich_2011-06.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))



from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print("==================================")
print(all_splits[1])
print("==================================")

print(len(all_splits))
















