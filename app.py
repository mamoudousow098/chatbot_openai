# langchain imports
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import  SentenceTransformerEmbeddings

## Transformers
from sentence_transformers import SentenceTransformer

import os
import json
from flask import Flask, jsonify, request
app = Flask(__name__)


def qa_chatbot(query) :
    path = "./data/"

    os.environ["OPENAI_API_KEY"] = "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"


    ## load documents
    loader = TextLoader(path  + "historique.txt")
    documents = loader.load()

    ## split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)


    ## select which embeddings we want to use
    embeddings = SentenceTransformerEmbeddings(model_name="Sahajtomar/french_semantic")

    ## create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)


    ## expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

    ## create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)


    try :
        result = qa({"query": query})
    except Exception as inst :
        result  = " nous sommes désoles mais il y'a un probléme avec l'api key de open api verifier votre compte"

    return result

@app.route('/chatbot', methods=['POST'])
def question_answers():

    data = json.loads(request.data)
    user_input = data['query']
    print(user_input)
    if user_input :
        answer = qa_chatbot(user_input)
        return jsonify({"answer" : answer }), 200




if __name__ == '__main__':
    app.run(port=5000, use_reloader=True)