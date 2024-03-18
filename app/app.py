import inspect
import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import pypdf
import streamlit as st
import streamlit_ext as ste
import tiktoken
import torch
from chromadb.errors import InvalidDimensionException
from datasets import Dataset
from langchain.callbacks.manager import CallbackManagerForChainRun, Callbacks
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (HuggingFaceInstructEmbeddings,
                                  OpenAIEmbeddings)
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import (BM25Retriever, EnsembleRetriever,
                                  MultiQueryRetriever)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai.chat_models import ChatOpenAI
from pydantic import Field
from ragas import evaluate
from ragas.metrics import (answer_correctness, answer_relevancy,
                           context_precision, context_recall, faithfulness)
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator
from sentence_transformers import CrossEncoder
from streamlit_chat import message

from prompts import HYDE_PROMPT, QA_CHAIN_PROMPT, QUESTION_GENERATOR_PROMPT

# Adjust the path to locate the config.py file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from config import DOCS_PATH, TESTSET_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"

# Functions 

def get_chunks(uploaded_docs, chunk_size, chunk_overlap):
    """
    Loads and processes uploaded documents.
    @param uploaded_docs: Set of uploaded documents.
    @return chunks: list of chunks.
    """
    combined_text = []
    for file in uploaded_docs:
        contents = file.read()
        # PDF file
        if file.type == "application/pdf":
            pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            combined_text.append(text)
        # Text file
        elif file.type == "text/plain":
            combined_text.append(contents.decode())
        else:
            st.warning("Unsupported file type for file: {}".format(file.name))
    text = " ".join(combined_text)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_len,
    )
    texts = text_splitter.split_text(text)
    chunks = text_splitter.create_documents(texts)

    return chunks


def get_bdp_documents(folder_path=DOCS_PATH):
    """
    Loads and processes Bank Customer Website documents from a specific folder.
    @param folder_path: The path of the folder containing the documents.
    @return documents: list of processed Document objects.
    """
    documents = []

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            json_data = read_bdp_json(file_path)
            if json_data:
                for entry in json_data.get("content", []):
                    doc = enrich_document(entry)
                    if doc:
                        documents.append(doc)

    # Update level 2 documents with corresponding level 3 topics
    update_level_2_documents(documents)

    return documents


def token_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def read_bdp_json(file_path):
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


def enrich_document(entry):
    """
    Creates and enriches a Document object from a JSON entry.
    @param entry: The JSON entry containing document data.
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
        page_content=page_content, metadata={"source": source, "topic": topic}
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


def get_bdp_chunks(documents, chunk_size, chunk_overlap):
    """
    Split documents into chunks
    @param text: documents to split
    @param chunk_size: charecters per split
    @param chunk_overlap: charecter overlap between chunks
    @return chunks: list of chunks
    """

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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

            # Add the chunk
            new_chunk = Document(
                page_content=chunked_text.page_content, metadata=chunked_metadata
            )
            chunks.append(new_chunk)

    return chunks


@st.cache_resource
def get_llm(model):
    """
    Load llm to generate answer
    @param model: llm model name
    @return llm: instance of the llm
    """

    if model == "GPT-4-turbo":
        llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)

    elif model == "Mistral-7B-Instruct-v0.1":
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            temperature=0.001,
            token=HUGGINGFACEHUB_API_TOKEN,
        )

    return llm


def get_retriever(
    chunks,
    embedding_model,
    query_transform,
    retriever_type,
    k_retrieved_docs,
    reranker_model,
    llm,
):
    """
    Get document retriever
    @param chunks: list of Documents (chunks)
    @param embedding_model: embedding model
    @param query_transform: method to transform the input query for retrieval
    @param retriever_type: retriever method
    @param reranker_model: reranker model
    @param k_retrieved_docs: number of retrieved chunks
    @param llm: llm
    @return: retriever
    """

    if embedding_model == "OpenAI":
        embd = OpenAIEmbeddings(model="text-embedding-3-small")

    elif embedding_model == "bge-large-en-v1.5":
        embd = HuggingFaceInstructEmbeddings(
            model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": device}
        )

    if reranker_model == "None":
        k = k_retrieved_docs
    else:
        k = k_retrieved_docs + 5

    if retriever_type == "Hybrid-search":
        try:
            vectorstore = Chroma.from_documents(chunks, embd)
        except (InvalidDimensionException, IndexError):
            Chroma().delete_collection()
            vectorstore = Chroma.from_documents(chunks, embd)
        chroma_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        bm25_retriever = BM25Retriever.from_documents(chunks, k=k)
        init_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
        )

    else:
        try:
            vectorstore = Chroma.from_documents(chunks, embd)
        except (InvalidDimensionException, IndexError):
            Chroma().delete_collection()
            vectorstore = Chroma.from_documents(chunks, embd)
        init_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

    if query_transform == "Multi-query":
        retriever = MultiQueryRetriever.from_llm(
            include_original=True, retriever=init_retriever, llm=llm
        )
    else:
        retriever = init_retriever

    return retriever


def reranking(reranker_model, retrieved_docs, k_retrieved_docs, question):
    """
    Rerank retrieved chunks
    @param reranker_model: reranker model to be used
    @param retrieved_docs: chunks coming from the retrieval step to be reranked
    @param question: question used on the retrieval step
    @return: reranked_docs
    """

    if reranker_model == "bge-reranker-large":
        model_name = "BAAI/bge-reranker-large"
        cross_encoder = CrossEncoder(model_name, device=device)
        scores = cross_encoder.predict(
            [[question, doc.page_content] for doc in retrieved_docs]
        )
        reranked_docs = [
            doc
            for doc, score in sorted(
                zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True
            )[:k_retrieved_docs]
        ]

    elif reranker_model == "None":
        reranked_docs = retrieved_docs[:k_retrieved_docs]

    return reranked_docs


def generate_testset(chunks, num_eval_questions):
    """
    Generate test set if no ground_tuth is passed.
    @param chunks: list of chunks
    @param num_eval_questions: number of question/answer pairs to generate
    @return testset: list of dictionaries with keys answer, question and ground_truth answer
    """

    # generator with openai models
    generator = TestsetGenerator.with_openai()

    # question type distribution
    distributions = {simple: 0.5, multi_context: 0.4, reasoning: 0.1}

    # generate test set
    generated = generator.generate_with_langchain_docs(
        chunks, num_eval_questions, distributions
    )
    generated_dataset = generated.to_dataset()

    # Reorganize the data into a list of dictionaries
    generated_dict = generated_dataset.to_dict()
    testset = [
        {key: value[i] for key, value in generated_dict.items()}
        for i in range(len(generated_dataset))
    ]

    return testset


def run_evaluation(
    chain, query_transform, retriever_type, reranker_model, retriever, testset
):
    """
    Runs evaluation of RAG pipeline performance on a given test set.
    @param chain: model chain used for answering questions
    @param retriever_type: retrieval method
    @param reranker_model: reranker model
    @param retriever: retriever object
    @param testset: list of dictionaries containing questions and corresponding ground truth answers
    @return: list of latency for each question
    @return: results (test set + metrics) for each question
    """

    data = []
    latency = []
    for entry in testset:
        # Get data from testset
        question = entry["question"]
        true_answer = entry["ground_truth"]

        # Log latency
        start_time = time.time()

        if retriever_type != "None":
            if query_transform == "HyDE":
                hyde_chain = LLMChain(llm=llm, prompt=HYDE_PROMPT, output_key="query")
                output = hyde_chain(question)
                query = output["query"]
            else:
                query = question

            retrieved_docs = retriever.get_relevant_documents(question)

            reranked_docs = reranking(
                reranker_model, retrieved_docs, k_retrieved_docs, question
            )

            # contexts.append([docs.page_content for docs in reranked_docs])
            context = "\n\n".join([doc.page_content for doc in reranked_docs])
            contexts = [docs.page_content for docs in reranked_docs]

        else:
            context = ""
            contexts = [context]

        # Generated Answer
        result = chain.invoke({"question": question, "context": context})

        # Compute latency
        end_time = time.time()
        elapsed_time = end_time - start_time
        latency.append(elapsed_time)

        # Evaluation set
        data.append(
            {
                "question": question,
                "answer": result["text"],
                "contexts": contexts,
                "ground_truths": [true_answer],
            }
        )

    # Convert dict to dataset
    data_df = pd.DataFrame(data)  
    dataset = Dataset.from_pandas(data_df)  

    # Get Results
    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ],
    )

    return latency, results


# Retriever for query transformation HyDE
class CustomRetriever(VectorStoreRetriever):
    def __init__(self, vectorstore: VectorStoreRetriever, chain: LLMChain, **kwargs):
        super().__init__(vectorstore=vectorstore, **kwargs)
        self.vectorstore = vectorstore
        self.chain = chain

    def get_relevant_documents(self, query: str) -> List[Document]:
        output = self.chain.invoke(query)
        new_query = output["new_question"] + "\n" + output["query"]
        results = self.vectorstore.get_relevant_documents(query=new_query)
        return results


# Conversational Retriever Chain with query transformation and reranking
class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
    reranker_model: Optional[str] = Field(
        default=None, description="The type of reranker to use"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        callbacks = _run_manager.get_child()
        new_question = self.question_generator.run(
            question=question, chat_history=chat_history_str, callbacks=callbacks
        )
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(
                new_question, inputs, self.reranker_model, run_manager=_run_manager
            )
        else:
            docs = self._get_docs(new_question, inputs, self.reranker_model)  # type: ignore[call-arg]
        output: Dict[str, Any] = {}
        if self.response_if_no_docs_found is not None and len(docs) == 0:
            output[self.output_key] = self.response_if_no_docs_found
        else:
            new_inputs = inputs.copy()
            if self.rephrase_question:
                new_inputs["question"] = new_question
            new_inputs["chat_history"] = chat_history_str
            answer = self.combine_docs_chain.run(
                input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs
            )
            output[self.output_key] = answer

        if self.return_source_documents:
            output["source_documents"] = docs
        if self.return_generated_question:
            output["generated_question"] = new_question
        return output

    def _get_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        reranker_model: str,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        retrieved_docs = self.retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        docs = reranking(
            self.reranker_model, retrieved_docs, k_retrieved_docs, question
        )
        return self._reduce_tokens_below_limit(docs)


st.set_page_config(page_title="🔍 RAG Chatbot Evaluation Tool 🛠️", layout="wide")

# App Design
st.info(
    """
    # 🤖 Q&A Chatbot Evaluation Tool

    **This is an evaluation tool for question-answering chatbots.** 💡

    - 📝 Create your custom RAG chatbot for your specific documents.
    - 🕵️ Evaluate the RAG pipeline for its quality in retrieval and question answering.
    - 🧩 Test with different settings to build your perfect chatbot assistant.

    ## How It Works:
    - **Document Selection**: Upload your documents or use the pre-loaded case. 📂
    - **Test set Options**: Submit your own test set or let the app generate it for convenience. 🎲
    - **Compare Different Experiments**: Analyse and compare the results of various experiments to find the best RAG settings. 🔎
    - **Launch Your Custom Chatbot**: Find the ideal RAG pipeline, launch your custom chatbot, and interact with it live! 🚀

    ## Metrics Used:
    The application utilizes the [RAGAs](https://docs.ragas.io/en/latest/index.html) (Retrieval-Augmented Generation Assessment) framework. The metrics employed, each ranging from 0 to 1, include:

    - **Context Precision**: Measures the signal-to-noise ratio of the retrieved context, computed using the question and its contexts. 🎯
    - **Context Recall**: Determines if all relevant information required to answer the question was retrieved. 📚
    - **Faithfulness**: Assesses the factual accuracy of the generated answer. 🛡️
    - **Answer Relevancy**: Evaluates how relevant the generated answer is to the posed question. 🔑
    - **Answer Correctness**: Compares the accuracy of the generated answer against the ground truth. ✅

    With this tool, you can experiment with different configurations and decide what best suits your data to create your perfect Chatbot! 📈
"""
)


# Chatbot parameters
with st.sidebar.form("parameters"):
    oai_api_key = st.text_input("`OpenAI API Key:`", type="password").strip()

    hf_api_key = st.text_input("`HuggingFace API Key:`", type="password").strip()

    chunk_size = st.select_slider(
        "`Chunk size (in tokens)`", options=[128, 256, 512, 768], value=256
    )

    chunk_overlap = st.select_slider(
        "`Chunk overlap (in tokens)`", options=[0, 20, 50, 100], value=20
    )

    query_transform = st.radio(
        "`Query Transformation`", ("None", "Multi-query", "HyDE"), index=0
    )

    embedding_model = st.radio(
        "`Embedding model`", ("bge-large-en-v1.5", "OpenAI"), index=0
    )

    retriever_type = st.radio(
        "`Retriever`", ("None", "Simple", "Hybrid-search"), index=1
    )

    k_retrieved_docs = st.select_slider(
        "`Retrieved chunks`", options=[1, 2, 3, 4, 5, 6], value=3
    )

    reranker_model = st.radio("`Reranker`", ("None", "bge-reranker-large"), index=0)

    model = st.radio(
        "`Chatbot model`",
        ("Mistral-7B-Instruct-v0.1", "GPT-4-turbo"),
        index=0,
        key="model",
    )

    submit_parameters_clicked = st.form_submit_button("Submit parameters")
    if submit_parameters_clicked:
        st.session_state.parameters_selected = True

    # Define API Keys
    HUGGINGFACEHUB_API_TOKEN = hf_api_key
    OPENAI_API_KEY = oai_api_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
    os.environ["OPENAI_API_KEY"] = oai_api_key

column_names = [
    "chunk size",
    "chunk overlap",
    "embedding model",
    "query transformation",
    "retriever",
    "k chunks retrieved",
    "reranker",
    "llm",
    "Context Precision",
    "Context Recall",
    "Faithfulness",
    "Answer Relevancy",
    "Answer Correctness",
    "Latency",
]

# Initialize session state variables if they don't exist
default_values = {
    "existing_df": pd.DataFrame(columns=column_names),
    "parameters_selected": False,
    "documents_selected": False,
    "uploaded_docs": None,
    "use_bp_example": None,
    "testset_option": None,
    "testset_selected": False,
    "documents_processed": False,
    "chatbot_active": False,
    "chatbot_loaded": False,
    "prompt": None,
    "show_experiment_results": False,
    "run_experiment": False,
    "detailed_df": pd.DataFrame(),
    "chunks": None,
    "user_input": "",
    "chat_history": [],
}


for key, default_value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

st.subheader("Document Upload")
st.session_state.uploaded_docs = st.file_uploader(
    "Upload documents (.pdf or .txt):", type=["pdf", "txt"], accept_multiple_files=True
)
st.session_state.use_bp_example = st.checkbox(
    "Use Banco de Portugal - Bank Customer Website"
)

# App options - default is launch chatbot
run_options = ["Launch Chatbot", "Run Experiment"]
st.session_state.run_option = st.radio("Select:", run_options)

if st.session_state.run_option == "Run Experiment":
    st.session_state.run_experiment = True
    st.session_state.show_experiment_results = True
    st.session_state.testset_selected = False

    # Delete current chatbot
    st.session_state.chatbot_active = False
    st.session_state.chat_history = []

    # Test Set Section
    if not st.session_state.testset_selected:
        st.subheader("Test Set")
        testset_options = ["Generate test set", "Upload test set"]
        if st.session_state.use_bp_example:
            testset_options.append("Use Bank Customer Website test set")

        st.session_state.testset_option = st.radio("Choose test set:", testset_options)

        if st.session_state.testset_option == "Generate test set":
            num_eval_questions = st.slider(
                "`Number of test questions (if generating test set)`",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
            )

        if st.session_state.testset_option == "Upload test set":
            uploaded_testset = st.file_uploader("Upload test set:", type="json")

            st.caption(
                """
            📋 Your test set must be a JSON file, containing a list of dictionaries. Each dictionary should represent a question-answer pair with two keys: `question` and `ground_truth`.
            The `question` key should map to the question you wish to include, and the `ground_truth` key to the corresponding answer. Here's the format:

            ```json
            [
              {
                "question": "Your question here",
                "ground_truth": "The corresponding answer to the question here"
              },
              {
                "question": "Another question here",
                "ground_truth": "The corresponding answer here"
              }
              // Add more question-answer pairs as needed
            ]
            """
            )

if st.session_state.run_option == "Launch Chatbot":
    st.session_state.chatbot_active = True
    st.session_state.run_experiment = False
    st.session_state.show_experiment_results = False

submit_clicked = st.button("Submit Selections")
if submit_clicked:
    st.session_state.documents_selected = True
    st.session_state.documents_processed = False

    if st.session_state.run_experiment == True:
        st.session_state.testset_selected = True

# Process documents if not already
if (
    not st.session_state.documents_processed
    and st.session_state.documents_selected
    and st.session_state.parameters_selected
):
    with st.spinner("Processing Documents..."):
        if st.session_state.uploaded_docs:
            st.session_state.chunks = get_chunks(
                st.session_state.uploaded_docs, chunk_size, chunk_overlap
            )
        elif st.session_state.use_bp_example:
            documents = get_bdp_documents()
            st.session_state.chunks = get_bdp_chunks(
                documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

        st.session_state.documents_processed = True


# RUN EXPERIMENT
if (
    st.session_state.run_experiment
    and st.session_state.documents_selected
    and st.session_state.testset_selected
):
    with st.spinner("Running Experiment..."):
        # Read Test Set
        if st.session_state.testset_option == "Upload test set":
            full_data = json.load(uploaded_testset)
            testset = [
                {"question": item["question"], "ground_truth": item["ground_truth"]}
                for item in full_data
                if "question" in item and "ground_truth" in item
            ]

        elif st.session_state.testset_option == "Generate test set":
            testset = generate_testset(st.session_state.chunks, num_eval_questions)

        elif st.session_state.testset_option == "Use Bank Customer Website test set":
            testset_path = TESTSET_PATH
            with open(testset_path, "r") as json_file:
                testset = json.load(json_file)

        # Get Necessary Objects
        llm = get_llm(model)
        retriever = get_retriever(
            st.session_state.chunks,
            embedding_model,
            query_transform,
            retriever_type,
            k_retrieved_docs,
            reranker_model,
            llm,
        )
        qa_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

        # Get Results
        st.session_state.latency, st.session_state.results = run_evaluation(
            qa_chain,
            query_transform,
            retriever_type,
            reranker_model,
            retriever,
            testset,
        )

if (
    st.session_state.show_experiment_results
    and st.session_state.documents_selected
    and st.session_state.testset_selected
):
    # Present Information
    tab1, tab2, tab3 = st.tabs(["RAG Results", "Question Detail", "Test set"])

    # Assemble outputs
    df = st.session_state.results.to_pandas()

    def format_context(context_list):
        formatted_context = ""
        # Iterate over each string in the context list
        for i, context in enumerate(context_list, start=1):
            formatted_context += f"Context {i}: {context}\n\n"

        return formatted_context.strip()

    # Apply the function to each item in the 'context' column
    df["contexts"] = df["contexts"].apply(format_context)

    df["latency"] = st.session_state.latency
    st.session_state.df = df

    mean_latency = df["latency"].fillna(0).mean()
    avg_context_precision = df["context_precision"].fillna(0).mean()
    avg_context_recall = df["context_recall"].fillna(0).mean()
    avg_answer_faithfulness = df["faithfulness"].fillna(0).mean()
    avg_answer_relavancy = df["answer_relevancy"].fillna(0).mean()
    avg_answer_correctness = df["answer_correctness"].fillna(0).mean()

    new_row = {
        "chunk size": chunk_size,
        "chunk overlap": chunk_overlap,
        "embedding model": embedding_model,
        "query transformation": query_transform,
        "retriever": retriever_type,
        "k chunks retrieved": k_retrieved_docs,
        "reranker": reranker_model,
        "llm": model,
        "Context Precision": avg_context_precision,
        "Context Recall": avg_context_recall,
        "Faithfulness": avg_answer_faithfulness,
        "Answer Relevancy": avg_answer_relavancy,
        "Answer Correctness": avg_answer_correctness,
        "Latency": mean_latency,
    }

    # Append the new experiment data to the existing DataFrame
    new_row_df = pd.DataFrame([new_row])
    st.session_state.existing_df = pd.concat(
        [st.session_state.existing_df, new_row_df], ignore_index=True
    )

    # Define columns for 'Parameters' and 'Results'
    parameters_columns = [
        "chunk size",
        "chunk overlap",
        "embedding model",
        "query transformation",
        "retriever",
        "k chunks retrieved",
        "reranker",
        "llm",
    ]
    results_columns = [
        "Context Precision",
        "Context Recall",
        "Faithfulness",
        "Answer Relevancy",
        "Answer Correctness",
        "Latency",
    ]

    # Tab 1: Aggregate Results of All Experiments
    with tab1:
        st.subheader("`Experiment Parameters`")
        parameters_df_all = st.session_state.existing_df[parameters_columns]
        st.dataframe(parameters_df_all, use_container_width=True)

        # Colorise table
        def colorize(value):
            if value >= 0.75 and value <= 1:
                color = "green"
            elif value >= 0.5 and value < 0.75:
                color = "yellow"
            elif value >= 0.25 and value < 0.5:
                color = "orange"
            else:  # Covers 0 to 0.25 and any other values (e.g., if not between 0 and 1)
                color = "red"
            return f"background-color: {color}"

        columns_to_color = [
            "Context Precision",
            "Context Recall",
            "Faithfulness",
            "Answer Relevancy",
            "Answer Correctness",
        ]
        results_df = st.session_state.existing_df[results_columns].style.applymap(
            colorize, subset=columns_to_color
        )

        st.subheader("`Experiment Results`")
        st.dataframe(results_df, use_container_width=True)

        # Dataframe for visualization
        show = st.session_state.existing_df.reset_index()
        show.columns = [
            "Experiment number",
            "chunk size",
            "chunk overlap",
            "embedding_model",
            "query transformation",
            "retriever",
            "k chunks retrieved",
            "reranker",
            "llm",
            "Context Precision",
            "Context Recall",
            "Faithfulness",
            "Answer Relevancy",
            "Answer Correctness",
            "Latency",
        ]
        show["Experiment number"] = show["Experiment number"].apply(
            lambda x: "Experiment #: " + str(x + 1)
        )

        # Radar Graph
        def melt_data_for_radar(show):
            filtered_show = show[
                [
                    "Experiment number",
                    "Latency",
                    "Context Precision",
                    "Context Recall",
                    "Faithfulness",
                    "Answer Relevancy",
                    "Answer Correctness",
                ]
            ]
            return filtered_show.melt(
                id_vars=["Experiment number", "Latency"],
                var_name="Metric",
                value_name="Value",
            )

        df_melted = melt_data_for_radar(show)
        fig1 = px.line_polar(
            df_melted,
            r="Value",
            theta="Metric",
            color="Experiment number",
            line_close=True,
            hover_data=["Latency"],
        )
        fig1.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
            ),
            showlegend=True,
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Tab 2: Detailed Results of the Last Experiment
    with tab2:
        st.subheader("`Parameters`")
        # Parameters of the last experiment
        parameters_df_last = st.session_state.existing_df.iloc[[-1]][parameters_columns]
        st.dataframe(parameters_df_last, use_container_width=True)

        st.subheader("`Question Details`")
        detail_columns = [
            "question",
            "answer",
            "ground_truth",
            "contexts",
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "latency",
        ]
        detail_df = df[detail_columns]
        columns_to_color = [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        ]
        color_df = detail_df.style.applymap(colorize, subset=columns_to_color)
        st.dataframe(color_df, use_container_width=False)

    # Display in Tab 3: Test Set
    with tab3:
        st.subheader("Test Set")
        st.dataframe(testset)

        # Download button for test set data
        json_content = json.dumps(testset)
        ste.download_button(
            label="Download test set JSON",
            data=json_content,
            file_name="YOUR_TESTSET.json",
            mime="application/json",
        )

# RUN CHATBOT #
if st.session_state.chatbot_active and st.session_state.documents_selected:
    if st.session_state.chatbot_loaded == False:
        with st.spinner("Loading Chatbot..."):
            st.session_state.prompt = QA_CHAIN_PROMPT 
            if "llm" not in st.session_state:
                st.session_state.llm = get_llm(model)
            if "retriever" not in st.session_state:
                st.session_state.retriever = get_retriever(
                    st.session_state.chunks,
                    embedding_model,
                    query_transform,
                    retriever_type,
                    k_retrieved_docs,
                    reranker_model,
                    st.session_state.llm,
                )

            st.session_state.chatbot_loaded = True

    if st.session_state.chatbot_loaded == True:
        memory = ConversationBufferWindowMemory(
            k=4,
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True,
        )

        for chat in st.session_state.chat_history:
            memory.save_context({"question": chat["human"]}, {"answer": chat["AI"]})

        chain = CustomConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": st.session_state.prompt},
            condense_question_prompt=QUESTION_GENERATOR_PROMPT,
            retriever=st.session_state.retriever,
            reranker_model=reranker_model,
            condense_question_llm=st.session_state.llm,
            chain_type="stuff",
            output_key="answer",
            response_if_no_docs_found="I am sorry, I don't have available information to answer that question.",
            verbose=True,
            return_generated_question=True,
            return_source_documents=True,
        )

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
