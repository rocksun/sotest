import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from llama_index.llms.gemini import Gemini


# def get_conversational_chain():

#     prompt_template = """
#     Answer the question as detailed as possible from the provided context with Chinese, 
#     make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", 
#     don't provide the wrong answer.\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash",
#                              temperature=0.1)

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain


def call_gemini(docs, user_question):
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    prompt_template = """
    Answer the question as detailed as possible from the provided context with Chinese, 
    make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", 
    don't provide the wrong answer.\n\n
    Context:\n\n{context}?\n
    Question: \n\n{question}\n
    Answer:
    """

    context = ""
    for doc in docs:
        context += doc.metadata["source"] + "\n\n"

    prompt = prompt_template.format(context=context, question=user_question)

    print(prompt)

    text = Gemini(model="models/gemini-1.5-flash", max_tokens=16192, safety_settings=safety_settings).complete(prompt).text
    return text



def docs_to_text(docs):
    text = ""
    for doc in docs:
        text += doc.metadata["source"] + "\n\n"
    return text

def user_input(user_question, new_db: FAISS):

#     prompt=f"""请从以下语句中提取名词，用空格分开，如果没有名词，就返回"没有名词"，不要返回其他信息：
# {user_question}
# """
#     entities = agent.run_prompt(prompt)
#     print(entities)
    
    # new_db = FAISS.load_local("faiss_index", embeddings)
    
    docs = new_db.similarity_search(user_question, k=2, fetch_k=4)
    print("matched:\n")
    # print(docs)

    # chain = get_conversational_chain()

    # response = chain(
    #     {"input_documents":docs, "question": user_question}
    #     , return_only_outputs=True)

    # print(response)

    # res = call_gemini(docs, user_question)
    # st.write("Reply: ", response["output_text"])
    # st.write("Reply: ", res)

    out = docs_to_text(docs)
    print(out)
    st.write("Reply: ", out)


def main(dir):
    db_path = os.path.join(dir,  "db")
    print(f"using db {db_path}")
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    st.set_page_config("Chat PDF")
    st.header("我的银行业务助手")

    user_question = st.text_input("询问银行相关的问题")

    if user_question:
        user_input(user_question, new_db)

    # with st.sidebar:
    #     st.title("Menu:")
    #     pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    #     if st.button("Submit & Process"):
    #         with st.spinner("Processing..."):
    #             raw_text = get_pdf_text(pdf_docs)
    #             text_chunks = get_text_chunks(raw_text)
    #             get_vector_store(text_chunks)
    #             st.success("Done")



if __name__ == "__main__":
    main("D:\\learns\\dcits\\test\\text")