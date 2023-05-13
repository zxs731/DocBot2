from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.prompt import PromptTemplate
import os
from flask import Flask, request, jsonify,render_template

BASE_URL = os.environ.get('OPENAI_API_BASE')
API_KEY = os.environ.get('OPENAI_API_KEY')
Version = os.environ.get('OPENAI_API_VERSION')
DEPLOYMENT_NAME = "chatgpt0301"

app = Flask(__name__)
print("initial embedding...")
#setup embedding db and build vectorstores
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

db_name='qa_db'
#vectorstore = None
embeddings = OpenAIEmbeddings(
        deployment="embedding"
    )
def InitialDB():
    #global vectordb
    with open('state_of_the_union.txt') as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([state_of_the_union])
    
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    for i in range(0,len(texts)):
        vectorstore.add_texts([texts[i].page_content])
    vectorstore.persist()
    vectorstore=None


InitialDB()
print("initial llm and chains...")
vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)

@app.route('/loaddb', methods=['POST'])
def LoadDB():
    global vectorstore
    global db_name
    global qa
    data = request.get_json()
    db_name = data['dbname']
    print("change and load db to "+db_name+"...")
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    vectorstore.persist()
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())


@app.route('/embedding', methods=['POST'])
def Embedding():
    global vectorstore
    global db_name
    global qa
    data = request.get_json()
    db_name = data['dbname']
    text = data['text']
    print("embedding and load db to "+db_name+"...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([text])
    
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    for i in range(0,len(texts)):
        vectorstore.add_texts([texts[i].page_content])
    vectorstore.persist()
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

    


llm = AzureChatOpenAI(
    temperature=0,
    model_name="gpt-35-turbo",
    openai_api_base=BASE_URL,
    openai_api_version="2023-03-15-preview",
    deployment_name="chatgpt0301",
    openai_api_key=API_KEY,
    openai_api_type = "azure",
    verbose=True,
)
# define two LLM models from OpenAI    
streaming_llm = AzureChatOpenAI(
    temperature=0.2,
    model_name="gpt-35-turbo",
    openai_api_base=BASE_URL,
    openai_api_version="2023-03-15-preview",
    deployment_name="chatgpt0301",
    openai_api_key=API_KEY,
    openai_api_type = "azure",
    streaming=True,
    callback_manager=CallbackManager([
        StreamingStdOutCallbackHandler()
    ]),
    verbose=True,
    max_tokens=150
)
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

@app.route('/answer', methods=['POST'])
def answer():
    print("get answer...")
    data = request.get_json()
    question = data['question']
    histories = data['histories']
    his=[]
    for i in histories:
        his.append((i["human"],i["AI"]))

    
    r = qa({"question": question,"chat_history":his })
    return jsonify({'answer': r['answer']})

@app.route('/db', methods=['GET'])
def db():
   return jsonify({'db_name': db_name})

@app.route('/')
def index():
    return render_template("bot.html")

if __name__ == '__main__':
    app.run(debug=True)