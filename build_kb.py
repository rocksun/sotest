from sotest import embed
import os


    # find all pdf file in dir and all child dirs
def store_all_text_file(dir):
    db_path = os.path.join(dir,  "db")
    chunks = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".txt"):
                text_file = os.path.join(root, file)
                print(f"process text file {text_file}")
                chunks = chunks + embed.get_qa_chunks_from_file(text_file)
    docs = embed.remove_duplicated_chunks(chunks)

    print(f"save chunks to {db_path}")
    for doc in docs:
        print(docs)
    # print(chunks)
    embed.save_qa_chunks_to_vector_store(docs, db_path)


if __name__ == "__main__":
    store_all_text_file("D:\\learns\\dcits\\test\\text")
    # main()
    # md_path = "D:\\learns\\dcits\\director\\day7-深农商非功能需求设计方案V0.1\\深农商非功能需求设计方案V0.1.pdf.origin.md"
    # embed.store_markdown_file(md_path)