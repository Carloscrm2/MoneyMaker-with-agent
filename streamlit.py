import streamlit as st

from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client

from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.utilities import SerpAPIWrapper
from langchain.tools import Tool

from utils import *
import os

os.environ['OPENAI_API_KEY'] = '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'
os.environ['SERPAPI_API_KEY'] = '47429bacc8821065f16542ed570574136b568424255b4abb7fc63b316b2f5a53'

client = Client()

st.set_page_config(
    page_title="ChatLangChain",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Chat🦜🔗"


@st.cache_resource(ttl="1h")
def configure_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = load_db(embedding_function=embeddings, persistence_directory='demodb').as_retriever(search_type = 'similarity', search_kwargs={"k": 3})
    
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
    compressor = LLMChainExtractor.from_llm(llm=llm)
    compression_retriver = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vectorstore)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriver, chain_type="stuff")

    # return vectorstore.as_retriever(search_type = 'similarity', search_kwargs={"k": 3})
    return qa

# tool = create_retriever_tool(
#     configure_retriever(),
#     "search_langsmith_docs",
#     "Searches and returns documents regarding LangSmith. LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications. You do not know anything about LangSmith, so if you are ever asked about LangSmith you should use this tool.",
# )
search = SerpAPIWrapper()
tools = [
    Tool(
        name='Document_Store',
        func=configure_retriever().run,
        description="Use it to lookup information from the document store. Always used as first tool."
    ),
    Tool(
        name='Search',
        func=search.run,
        description='Use this to lookup information from googel search engine. Use it only after you have tried using the Document_Store tool.'
    )
         ]
llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")
memory = AgentTokenBufferMemory(memory_key="history", llm=llm)
message = SystemMessage(
    content=(
    """Answer the following questions as best you can. \
    You have access to the following tools:

    Document_Store: Use it to lookup information from document store. \
                    Always used as first tool
    Search: Use this to lookup information from google search engine. \
            Use it only after you have tried using the document store tool.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [Document_Store, Search]. \
            Always look first in Document_Store.
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question.

    Use three sentences maximum. Keep the answer as concise as possible.

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""
    )
)
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)
# memory = AgentTokenBufferMemory(llm=llm)
starter_message = "Ask me anything about finanance"
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]


def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)


for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id

        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))