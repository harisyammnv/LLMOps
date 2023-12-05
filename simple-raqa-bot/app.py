from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
)
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
import os
import openai
import chainlit as cl
import asyncio
from dotenv import load_dotenv
from chainlit.types import AskFileResponse
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyMuPDFLoader
import lancedb
from langchain.vectorstores.lancedb import LanceDB 
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from prompts import RAQA_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

welcome_message = """Welcome to the RAQA bot developed for LLMops3 !!! 
To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""
db = lancedb.connect("/tmp/lancedb")
embeddings = OpenAIEmbeddings()
table = db.create_table(
    "my_table",
    data=[
        {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }
    ],
    mode="overwrite",
)

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), 
                 vector_db_retriever: VectorDatabase,
                 raqa_prompt: str,
                 user_prompt: str) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever
        self.raqa_prompt = raqa_prompt
        self.user_prompt = user_prompt

    def run_pipeline(self, user_query: str) -> str:
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)
        
        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_system_prompt = self.raqa_prompt.create_message(context=context_prompt)

        formatted_user_prompt = self.user_prompt.create_message(user_query=user_query)
        
        return self.llm.run([formatted_system_prompt, formatted_user_prompt])


def process_docs(file: AskFileResponse):
    import tempfile

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tempfile:
        if file.type == "application/pdf":
            with open(tempfile.name, "wb") as f:
                f.write(file.content)
            
            pdf_loader = PyMuPDFLoader(tempfile.name)
            documents = pdf_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_documents = text_splitter.split_texts(documents)
            for i, doc in enumerate(split_documents):
                doc.metadata["source"] = f"source_{i}"
            vector_db = LanceDB.from_documents(documents, embeddings, connection=table)
            

        elif file.type == "text/plain":
            with open(tempfile.name+'.txt', "wb") as f:
                f.write(file.content)
            text_loader = TextFileLoader(tempfile.name+'.txt')
            documents = text_loader.load_documents()
            text_splitter = CharacterTextSplitter()
            split_documents = text_splitter.split_texts(documents)

            vector_db = VectorDatabase()
            vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))
    
    return vector_db



@cl.on_chat_start
async def on_chat_start():

    await cl.Avatar(
        name="Chatbot",
        path="icon/chainlit.png"
        ).send()
    
    await cl.Avatar(
        name="User",
        path="icon/avatar.png",
    ).send()
    
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
            disable_human_feedback=True,
        ).send()

    file = files[0]
    
    msg = cl.Message(
        content=f"Processing `{file.name}`...Please wait", disable_human_feedback=True
    )
    await msg.send()

    vector_db = await cl.make_async(process_docs)(file)

    

    if file.name.endswith(".pdf"):
        message_history = ChatMessageHistory()

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
            chain_type="stuff",
            retriever=vector_db.as_retriever(),
            memory=memory,
            return_source_documents=True,
        )
        cl.user_session.set("chain", chain)

    if file.name.endswith(".txt"):
        chat_openai = ChatOpenAI()
        raqa_prompt = SystemRolePrompt(RAQA_PROMPT_TEMPLATE)


        user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)

        retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
            vector_db_retriever=vector_db,
            llm=chat_openai,
            raqa_prompt=raqa_prompt,
            user_prompt=user_prompt
        )

        cl.user_session.set("pipeline", retrieval_augmented_qa_pipeline)

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    cl.user_session.set("file_name", file.name)
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    file_name = cl.user_session.get("file_name")
    if file_name.endswith(".txt"):
        pipeline = cl.user_session.get("pipeline")

        response = pipeline.run_pipeline(message.content)

        await cl.Message(content=response).send()
    if file_name.endswith(".pdf"):
        chain = cl.user_session.get("chain")
        cb = cl.AsyncLangchainCallbackHandler()
        res = await chain.acall(message.content, callbacks=[cb])
        answer = res["answer"]
        source_documents = res["source_documents"]

        text_elements = []

        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                # Create the text element referenced in the message
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]

            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"
    