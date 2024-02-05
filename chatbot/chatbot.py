import io
import json
import os
import sys

# Adjust the path to locate the config.py file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pypdf
import streamlit as st
import tiktoken
from auxiliary import (ANSWER_PROMPT_BDP, ANSWER_PROMPT_DEFAULT,
                       MULTIQUERY_PROMPT, QUERY_EXPANSION_PROMPT,
                       QUESTION_GENERATOR_PROMPT,
                       CustomConversationalRetrievalChain, CustomRetriever,
                       LineListOutputParser, device)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (HuggingFaceInstructEmbeddings,
                                  OpenAIEmbeddings)
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import (BM25Retriever, EnsembleRetriever,
                                  MultiQueryRetriever)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from streamlit_chat import message

from config import BASE_MISTRAL_PATH, DOCS_PATH, SMALL_MISTRAL_PATH

# Set Streamlit page configuration
st.set_page_config(page_title="🧠Custom ChatBot🤖", layout="wide")
st.header("Custom ChatBot🤖")
st.subheader("Powered by 🦜 LangChain + 👑 Streamlit")


def get_query_params():
    query_params = st.query_params

    params = {
        "oai_api_key": query_params.get("oai_api_key", "None"),
        "chunk_size": int(query_params.get("chunk_size", "256")),
        "chunk_overlap": int(query_params.get("chunk_overlap", "20")),
        "retriever_type": query_params.get("retriever_type", "Similarity"),
        "num_neighbors": int(query_params.get("num_neighbors", "3")),
        "hybrid_weight": float(query_params.get("hybrid_weight", "0.0")),
        "reranker_type": query_params.get("reranker_type", "LongContextReorder"),
        "embeddings": query_params.get("embeddings", "bge-base-en-v1.5"),
        "model": query_params.get("model", "Mistral-7B-Instruct-v0.1 Small"),
    }
    return params


# Retrieve and display parameters
params = get_query_params()
st.markdown("### 🛠️ Configured Parameters")
# Create a grid layout for better visualization
cols = st.columns(2)
for i, (key, value) in enumerate(params.items()):
    with cols[i % 2]:
        st.markdown(f"**{key.capitalize()}**: `{value}`")


# Assign parameters to variables
oai_api_key = params["oai_api_key"]
chunk_size = params["chunk_size"]
chunk_overlap = params["chunk_overlap"]
retriever_type = params["retriever_type"]
num_neighbors = params["num_neighbors"]
hybrid_weight = params["hybrid_weight"]
embeddings = params["embeddings"]
reranker_type = params["reranker_type"]
model = params["model"]


def read_json_file(file_path):
    """
    Reads a JSON file and returns its contents.
    @param file_path: The path of the file to read.
    @return: The contents of the JSON file or None if an error occurs.
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except IOError as e:
        st.error(f"Error reading file {file_path}: {e}")
        return None


def enrich_document(entry, doc_id):
    """
    Creates and enriches a Document object from a JSON entry.
    @param entry: The JSON entry containing document data.
    @param doc_id: The ID to assign to the document.
    @return: The enriched Document object or None if entry is invalid.
    """
    answer = entry.get("answer", "")
    if answer == "":
        return None
    source = entry.get("link", "")
    topic = entry.get("question") or entry.get("question_topic", "")
    if source == "https://clientebancario.bportugal.pt/en/perguntas-frequentes":
        topic_for_content = topic.split(" -> ")[-1]
        page_content = f"{topic_for_content}\n\n{answer}"
    else:
        page_content = f"{topic}\n\n{answer}"
    return Document(
        page_content=page_content,
        metadata={"source": source, "topic": topic, "doc_id": doc_id},
    )


def update_level_2_documents(documents):
    """
    Updates level 2 documents with corresponding level 3 topics.
    @param documents: A list of Document objects.
    @return: None. The function modifies the documents list in place.
    """
    level_2_docs = {}
    level_3_topics = []

    # Categorize documents based on their level
    for doc in documents:
        topic = doc.metadata.get("topic", "")
        level = topic.count("->") + 1
        if level == 2:
            level_2_docs[topic] = doc
        elif level == 3:
            level_3_topics.append(topic)

    # Update level 2 documents if they have corresponding level 3 topics
    for level_2_topic, level_2_doc in level_2_docs.items():
        if level_2_doc.page_content.endswith(":"):
            # Extract the base of the level 2 topic for matching with level 3 topics
            base_topic = level_2_topic + " -> "
            matching_topics = [
                t.split(" -> ")[-1] for t in level_3_topics if t.startswith(base_topic)
            ]
            if matching_topics:
                level_2_doc.page_content += "\n" + "\n".join(matching_topics)


def get_documents(folder_path=DOCS_PATH):
    """
    Loads and processes documents from a specified folder.
    @param folder_path: The path of the folder containing the documents.
    @return: A list of processed Document objects.
    """
    documents = []
    doc_id = 0

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            json_data = read_json_file(file_path)
            if json_data:
                for entry in json_data.get("content", []):
                    doc = enrich_document(entry, doc_id)
                    if doc:
                        documents.append(doc)
                        doc_id += 1

    # Update level 2 documents with corresponding level 3 topics
    update_level_2_documents(documents)

    return documents


def token_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def get_chunks(documents, chunk_size, overlap):
    """
    Split documents into chunks
    @param text: documents to split
    @param chunk_size: charecters per split
    @param overlap: charecter overlap between chunks
    @return: list of chunks
    """

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=token_len,
    )
    chunks = []
    for doc in documents:
        page_content = doc.page_content
        metadata = doc.metadata
        chunked_doc = text_splitter.create_documents(
            texts=[page_content], metadatas=[metadata]
        )
        for chunked_text in chunked_doc:
            chunked_metadata = chunked_text.metadata
            # remove chunks that only have the topic or the FAQ
            if (
                chunked_text.metadata["source"]
                == "https://clientebancario.bportugal.pt/en/perguntas-frequentes"
            ):
                topic = chunked_text.metadata["topic"].split(" -> ")[-1]
            else:
                topic = chunked_text.metadata["topic"]
            if chunked_text.page_content.strip() == topic.strip():
                continue
            else:
                new_chunk = Document(
                    page_content=chunked_text.page_content, metadata=chunked_metadata
                )
                chunks.append(new_chunk)

    return chunks


def load_mistral(mistral_model):
    """
    Load mistral model from path
    @mistral_model: mistral model to load
    """

    if mistral_model == "Mistral-7B-Instruct-v0.1 Small":
        model_path = SMALL_MISTRAL_PATH

    elif mistral_model == "Mistral-7B-Instruct-v0.1 Med":
        model_path = BASE_MISTRAL_PATH

    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.0,
        top_p=1,
        n_ctx=4096,
        # uncomment if using GPU
        # n_batch=1024,
        # n_gpu_layers=100,
    )

    return llm


def get_llm(model):
    """
    Get LLM
    @param model: LLM to use
    @return: LLM
    """

    if model == "GPT-4-turbo":
        llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)

    elif (
        model == "Mistral-7B-Instruct-v0.1 Small"
        or model == "Mistral-7B-Instruct-v0.1 Med"
    ):
        llm = load_mistral(model)

    return llm


def get_retriever(
    chunks, retriever_type, embedding_type, num_neighbors, llm, hybrid_weight
):
    """
    Get document retriever
    @param chunks: list of Documents (chunks)
    @param retriever_type: retriever type
    @param embedding_type: embedding type
    @param num_neighbors: number of neighbors for retrieval
    @return: retriever
    """

    # Set embeddings
    if embedding_type == "OpenAI":
        embd = OpenAIEmbeddings(model="text-embedding-3-small")

    elif embedding_type in [
        "bge-base-en-v1.5",
        "bge-small-en-v1.5",
        "bge-large-en-v1.5",
    ]:
        model_name = f"BAAI/{embedding_type}"
        embd = HuggingFaceInstructEmbeddings(
            model_name=model_name, model_kwargs={"device": device}
        )

    # Select retriever
    if retriever_type == "Similarity":
        vectorstore = Chroma.from_documents(chunks, embd)
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": num_neighbors, "score_threshold": 0.4},
        )

    if retriever_type == "BM25":
        retriever = BM25Retriever.from_documents(chunks)

    elif retriever_type == "Hybrid-search":
        vectorstore = Chroma.from_documents(chunks, embd)
        chroma_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": num_neighbors, "score_threshold": 0.4},
        )
        bm25_retriever = BM25Retriever.from_documents(chunks)
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[hybrid_weight, 1 - hybrid_weight],
        )

    elif retriever_type == "Similarity + Expansion w/ Generated Answers":
        vectorstore = Chroma.from_documents(chunks, embd)
        chain = LLMChain(llm=llm, prompt=QUERY_EXPANSION_PROMPT, output_key="query")
        retriever = CustomRetriever(
            vectorstore=vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": num_neighbors, "score_threshold": 0.4},
            ),
            chain=chain,
        )
        mq_retrieval_chain = LLMChain(
            llm=llm, prompt=MULTIQUERY_PROMPT, output_parser=LineListOutputParser()
        )

    elif retriever_type == "MultiQuery Retriever":
        vectorstore = Chroma.from_documents(chunks, embd)
        chain = LLMChain(
            llm=llm, prompt=MULTIQUERY_PROMPT, output_parser=LineListOutputParser()
        )
        retriever = MultiQueryRetriever(
            retriever=vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": num_neighbors, "score_threshold": 0.4},
            ),
            llm_chain=mq_retrieval_chain,
            parser_key="lines",
            include_original=True,
        )

    return retriever


# Run App

if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Document Upload Section - Shown only if not submitted
if not st.session_state.submitted:
    st.subheader("Document Upload")
    uploaded_docs = st.file_uploader(
        "Upload documents (.pdf or .txt):",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    use_bp_example = st.checkbox("Use Banco de Portugal - Bank Customer Website")
    submit_selections = st.button("Submit Selections")
    if submit_selections:
        st.session_state.submitted = True
        st.session_state.uploaded_docs = uploaded_docs
        st.session_state.use_bp_example = use_bp_example

# Processing after document submission
if st.session_state.submitted:
    with st.spinner("Loading ChatBot..."):
        if st.session_state.use_bp_example:
            documents = get_documents()
            chunks = get_chunks(documents, chunk_size, chunk_overlap)
            prompt = ANSWER_PROMPT_BDP

        if st.session_state.uploaded_docs:
            combined_text, fnames = [], []
            for file in sorted(st.session_state.uploaded_docs):
                contents = file.read()
                # PDF file
                if file.type == "application/pdf":
                    pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    combined_text.append(text)
                    fnames.append(file.name)
                # Text file
                elif file.type == "text/plain":
                    combined_text.append(contents.decode())
                    fnames.append(file.name)
                else:
                    st.warning("Unsupported file type for file: {}".format(file.name))
            text = " ".join(combined_text)
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            texts = text_splitter.split_text(text)
            chunk_ids = [{"chunk_id": i} for i in range(len(texts))]
            chunks = text_splitter.create_documents(texts, chunk_ids)
            prompt = ANSWER_PROMPT_DEFAULT

        # Get LLM
        if "llm" not in st.session_state:
            st.session_state.llm = get_llm(model)

        # Get retriver
        if "retriever" not in st.session_state:
            st.session_state.retriever = get_retriever(
                chunks,
                retriever_type,
                embeddings,
                num_neighbors,
                st.session_state.llm,
                hybrid_weight,
            )

if "llm" in st.session_state:
    memory = ConversationBufferWindowMemory(
        k=4,
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    else:
        for chat in st.session_state.chat_history:
            memory.save_context({"question": chat["human"]}, {"answer": chat["AI"]})

    chain = CustomConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        condense_question_prompt=QUESTION_GENERATOR_PROMPT,
        retriever=st.session_state.retriever,
        reranker_type=reranker_type,
        condense_question_llm=st.session_state.llm,
        chain_type="stuff",
        output_key="answer",
        response_if_no_docs_found="I am sorry, I don't have available information to answer that question.",
        verbose=True,
        return_generated_question=True,
        return_source_documents=True,
    )

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    def submit():
        st.session_state.user_input = st.session_state.widget
        st.session_state.widget = ""

    st.text_input(
        "You: ",
        key="widget",
        placeholder="Hello, I am your AI assistant! How can I help you?",
        label_visibility="hidden",
        on_change=submit,
    )

    user_input = st.session_state.user_input

    if user_input:
        with st.spinner("Generating answer..."):
            output = chain.invoke(input=user_input)
            answer = output["answer"]
            sources = output.get("source_documents", [])

        # Only show sources for the current (last) output
        if sources:
            with st.expander("Sources used in Answer", expanded=False):
                for i, source in enumerate(sources, start=1):
                    st.markdown(f"**Source {i}:**\n{source.page_content}")

        st.session_state.chat_history.append({"human": user_input, "AI": answer})

    messages = st.session_state.get("chat_history", [])
    # Display messages in reverse order
    for msg in reversed(messages):
        if "human" in msg and msg["human"]:
            message(msg["human"], is_user=True, key=str(msg) + "_user")
        if "AI" in msg and msg["AI"]:
            message(msg["AI"], is_user=False, key=str(msg) + "_ai")
