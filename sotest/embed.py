from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import re

def get_paragraphs(text):
    """
    Extract paragraphs from text. Paragraphs are separated by two or more newlines.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of paragraphs
    """
    # Split text by one or more newlines
    lines = text.split('\n')
    
    paragraphs = []
    current_paragraph = []
    empty_line_count = 0
    
    for line in lines:
        if line.strip():  # Non-empty line
            current_paragraph.append("\n"+line)
            empty_line_count = 0
        else:  # Empty line
            empty_line_count += 1
            if empty_line_count >= 2 and current_paragraph:
                # Join lines in current paragraph and add to result
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
    
    # Don't forget the last paragraph
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return paragraphs

def get_qa_chunks_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        # 通过3个或更多的空行分隔文本
        text_chunks = get_paragraphs(text)
        print(len(text_chunks))
        # text_chunks = get_text_chunks(text)
        return text_chunks

def remove_duplicated_chunks(text_chunks):
    unique_titles = []
    documents = []
    for chunk in text_chunks:
        chunk = chunk.strip()



        title_line = chunk.split("\n")[0].strip()
        # extract title that after first . or 、

        # print("`"+chunk+"`")
        title =  re.split('[.、]', title_line, maxsplit=1)[1].strip()

        if title not in unique_titles:
            unique_titles.append(title)
            print("`"+title+"`")
            documents.append(Document(
                page_content=title,
                metadata={"source": chunk},
            ))

    # print(unique_chunks)
    return documents

def save_qa_chunks_to_vector_store(documents, store_path):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
    # vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(store_path)