import io
import json
import os
import random
import re
import sys
import time

# Adjust the path to locate the config.py file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import altair as alt
import numpy as np
import pandas as pd
import pypdf
import streamlit as st
import tiktoken
from auxiliary import (BINARY_ANSWER_PROMPT, MULTIQUERY_PROMPT,
                       QA_CHAIN_PROMPT, QUERY_EXPANSION_PROMPT,
                       SCORE_ANSWER_PROMPT, CustomRetriever,
                       LineListOutputParser, device)
from chromadb.errors import InvalidDimensionException
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (HuggingFaceInstructEmbeddings,
                                  OpenAIEmbeddings)
from langchain.evaluation.qa import QAEvalChain, QAGenerateChain
from langchain.output_parsers import RegexParser
from langchain.retrievers import (BM25Retriever, EnsembleRetriever,
                                  MultiQueryRetriever)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_transformers import LongContextReorder
from langchain_community.llms import LlamaCpp
from sentence_transformers import CrossEncoder

from config import (BASE_BGE_PATH, BASE_MISTRAL_PATH, CHATBOT_URL, DOCS_PATH,
                    GROUND_TRUTH_PATH, LARGE_BGE_PATH, RERANKER_PATH,
                    SMALL_BGE_PATH, SMALL_MISTRAL_PATH)

st.set_page_config(page_title="🔍 ChatBot Evaluation Tool 🛠️", layout="wide")

# Define the columns for the aggregate results table
aggregate_results_columns = [
    "answer model",
    "retriever",
    "embedding",
    "chunk size",
    "chunk overlap",
    "k docs retrieved",
    "reranker",
    "evaluation model",
    "evaluation questions",
    "Answer score",
    "Precision",
    "Recall",
    "Hit Rate",
    "MRR",
    "AP",
    "NDCG",
    "Latency",
]


if "existing_df" not in st.session_state:
    st.session_state.existing_df = pd.DataFrame(columns=aggregate_results_columns)
else:
    # If it exists, use the existing DataFrame
    summary = st.session_state.existing_df


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
    if source == "https://clientebancario.bportugal.pt/en/perguntas-frequentes":  # FAQs
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
    chunk_id = 0
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
                == "https://clientebancario.bportugal.pt/en/perguntas-frequentes"  # FAQs
            ):
                topic = chunked_text.metadata["topic"].split(" -> ")[-1]
            else:
                topic = chunked_text.metadata["topic"]

            if chunked_text.page_content.strip() == topic.strip():
                continue

            # Add the chunk_id to the metadata
            chunked_metadata["chunk_id"] = chunk_id
            new_chunk = Document(
                page_content=chunked_text.page_content, metadata=chunked_metadata
            )
            chunks.append(new_chunk)
            chunk_id += 1

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


@st.cache_resource
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

    if retriever_type == "Similarity":
        try:
            vectorstore = Chroma.from_documents(chunks, embd)
        except InvalidDimensionException:
            Chroma().delete_collection()
            vectorstore = Chroma.from_documents(chunks, embd)
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": num_neighbors, "score_threshold": 0.4},
        )

    elif retriever_type == "BM25":
        retriever = BM25Retriever.from_documents(chunks)

    elif retriever_type == "Hybrid-search":
        try:
            vectorstore = Chroma.from_documents(chunks, embd)
        except InvalidDimensionException:
            Chroma().delete_collection()
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
        try:
            vectorstore = Chroma.from_documents(chunks, embd)
        except InvalidDimensionException:
            Chroma().delete_collection()
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
        try:
            vectorstore = Chroma.from_documents(chunks, embd)
        except InvalidDimensionException:
            Chroma().delete_collection()
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


def reranking(reranker_type, retrieved_docs, question):
    """
    Rerank retrieved docs
    @param reranker_type: reranker model to be used
    @param retrieved_docs: documents coming from the retrieval step to be reranked
    @param question: question used on the retrieval step
    @return: retriever
    """
    reranked_docs = retrieved_docs

    if reranker_type in ["ms-marco-MiniLM-L-6-v2", "bge-reranker-base"]:
        model_name = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
            if reranker_type == "ms-marco-MiniLM-L-6-v2"
            else "BAAI/bge-reranker-base"
        )
        cross_encoder = CrossEncoder(model_name, device=device)
        scores = cross_encoder.predict(
            [[question, doc.page_content] for doc in retrieved_docs]
        )
        reranked_docs = [
            doc
            for doc, score in sorted(
                zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True
            )
        ]

    elif reranker_type == "LongContextReorder":
        reranker = LongContextReorder()
        reranked_docs = reranker.transform_documents(retrieved_docs)

    return reranked_docs


def generate_eval(chunks, llm, num_questions):
    """
    Generate multiple question/answer pairs from random chunks based on chunk_id.
    @param chunks: list of chunks with chunk_id
    @param num_questions: number of question/answer pairs to generate
    @return: list of dicts, each with keys "question", "answer" and "chunk_id"
    """

    qa_gen_chain = QAGenerateChain.from_llm(
        llm,
        output_parser=RegexParser(
            regex="QUESTION: (.*?)\\n+ANSWER: (.*)", output_keys=["question", "answer"]
        ),
    )
    eval_pairs = []

    for _ in range(num_questions):
        random_chunk = random.choice(chunks)
        chunk_text = random_chunk.page_content
        awaiting_answer = True
        while awaiting_answer:
            try:
                output = qa_gen_chain.invoke(chunk_text)
                qa_pair = output["qa_pairs"]
                qa_pair["sources"] = [random_chunk.metadata["chunk_id"]]
                eval_pairs.append(qa_pair)
                awaiting_answer = False
            except:
                st.error("Error on question")
                random_chunk = random.choice(chunks)
                chunk_text = random_chunk.page_content

    return eval_pairs


def grade_model_answer(gt_dataset, predictions, model_eval, grade_answer_prompt):
    """
    Grades the answer based on ground truth and model predictions.
    @param gt_dataset: list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @model_eval: Model used to evaluate the answers & retrieval
    @param grade_answer_prompt: The prompt for the grading. Either "Binary" or "Score".
    @return answers_grade: A list of strings - scores + reasoning for the generated answers.
    """

    if grade_answer_prompt == "Binary":
        prompt = BINARY_ANSWER_PROMPT
    elif grade_answer_prompt == "Score":
        prompt = SCORE_ANSWER_PROMPT

    if model_eval == "GPT-4-turbo":
        eval_chain = QAEvalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0),
            prompt=prompt,
        )

    elif (
        model == "Mistral-7B-Instruct-v0.1 Small"
        or model == "Mistral-7B-Instruct-v0.1 Med"
    ):
        llm = load_mistral(model)
        eval_chain = QAEvalChain.from_llm(llm=llm, prompt=prompt)

    answers_grade = eval_chain.evaluate(
        gt_dataset, predictions, question_key="question", prediction_key="result"
    )
    return answers_grade


def grade_model_retrieval(gt_dataset, retrieved_docs):
    """
    Calculate Retrieval Metrics for each question in the ground truth dataset.

    @param gt_dataset: Ground truth dataset, a list of dicts with 'question' and 'sources' (correct chunk_id sources for the question).
    @param retrieved_docs: Retrieved documents, a list of dicts with 'question' and 'reranked_doc_ids' (list of reranked chunk_ids for the question).

    @return retrieval_metrics: A list of dicts with the retrieval metrics for each question, including precision, recall, hit rate, MRR, AP, and NDCG.
    """

    # Convert gt_dataset into a more accessible format
    truth_dict = {item["question"]: set(item["sources"]) for item in gt_dataset}

    retrieval_metrics = []

    # Helper function to calculate NDCG
    def ndcg_at_k(relevance_scores, num_retrieved_docs, num_relevant_docs):
        r = np.asfarray(relevance_scores)[:num_retrieved_docs]
        if r.size:
            # Calculate DCG
            log_base = np.log2(np.arange(2, r.size + 2))
            dcg = np.sum(r / log_base)

            # Calculate IDCG
            ideal_relevance_scores = np.ones(min(num_relevant_docs, num_retrieved_docs))
            idcg_log_base = np.log2(
                np.arange(2, min(num_relevant_docs, num_retrieved_docs) + 2)
            )
            idcg = np.sum(ideal_relevance_scores / idcg_log_base)

            # Calculate NDCG
            ndcg_score = dcg / idcg if idcg else 0

            # Debugging output
            # st.write("DCG:", dcg, "IDCG:", idcg, "NDCG:", ndcg_score)
            return ndcg_score
        return 0

    for doc in retrieved_docs:
        question = doc["question"]
        reranked_doc_ids = doc["reranked_doc_ids"]
        true_chunk_ids = truth_dict.get(question, set())

        tp = len(true_chunk_ids.intersection(reranked_doc_ids))
        fp = len(set(reranked_doc_ids) - true_chunk_ids)
        fn = len(true_chunk_ids - set(reranked_doc_ids))

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        hit_rate = 1 if tp else 0

        relevant_ranks = [
            1 / (i + 1)
            for i, chunk_id in enumerate(reranked_doc_ids)
            if chunk_id in true_chunk_ids
        ]
        mrr = max(relevant_ranks, default=0)

        cum_tp = 0
        precisions = []
        for i, chunk_id in enumerate(reranked_doc_ids):
            if chunk_id in true_chunk_ids:
                cum_tp += 1
                precisions.append(cum_tp / (i + 1))
        ap = np.mean(precisions) if precisions else 0

        relevance_scores = [
            1 if chunk_id in true_chunk_ids else 0 for chunk_id in reranked_doc_ids
        ]
        num_retrieved_docs = len(reranked_doc_ids)
        num_relevant_docs = len(true_chunk_ids)
        ndcg = ndcg_at_k(relevance_scores, num_retrieved_docs, num_relevant_docs)

        retrieval_metrics.append(
            {
                "question": question,
                "precision": precision,
                "recall": recall,
                "Hit Rate": hit_rate,
                "MRR": mrr,
                "AP": ap,
                "NDCG": ndcg,
            }
        )

    return retrieval_metrics


def run_evaluation(chain, retriever, gt_dataset, model_eval, grade_prompt, text):
    """
    Runs evaluation on a model's performance on a given evaluation dataset.
    @param chain: Model chain used for answering questions
    @param retriever:  Document retriever used for retrieving relevant documents
    @param eval_set: List of dictionaries containing questions and corresponding ground truth answers
    @param model_eval: Model used to grade the answers & retrieval
    @param grade_prompt: String prompt used for grading model's performance
    @return: A tuple of four items:
    - answers_grade: A dictionary containing scores for the model's answers.
    - retrieval_metrics: A list of lists with the retrieval metrics for each question.
    - latencies_list: A list of latencies in seconds for each question answered.
    - predictions_list: A list of dictionaries containing the model's predicted answers and relevant documents for each question.
    """

    predictions = []
    retrieved_docs = []
    latencies_list = []

    for eval_qa_pair in gt_dataset:
        # Get answer and log latency
        start_time = time.time()
        question = eval_qa_pair["question"]

        # Retrieved docs
        initial_docs = retriever.get_relevant_documents(question)
        docs = reranking(reranker_type, initial_docs, question)
        reranked_doc_ids = [doc.metadata["chunk_id"] for doc in docs]
        retrieved_docs.append(
            {"question": question, "reranked_doc_ids": reranked_doc_ids}
        )

        # Prediction
        context = "\n\n".join([doc.page_content for doc in docs])
        result = chain.invoke({"question": question, "context": context})
        predictions.append(
            {
                "question": question,
                "answer": eval_qa_pair["answer"],
                "result": result["text"],
            }
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        latencies_list.append(elapsed_time)

    # Grading
    answers_grade = grade_model_answer(
        gt_dataset, predictions, model_eval, grade_prompt
    )
    retrieval_metrics = grade_model_retrieval(gt_dataset, retrieved_docs)

    return answers_grade, retrieval_metrics, latencies_list, predictions


def generate_url(
    oai_api_key,
    chunk_size,
    chunk_overlap,
    embeddings,
    retriever_type,
    hybrid_weight,
    num_neighbors,
    reranker_type,
    model,
):
    return f"{CHATBOT_URL}/?oai_api_key={oai_api_key.strip()}&chunk_size={chunk_size}&chunk_overlap={chunk_overlap}&retriever_type={retriever_type}&num_neighbors={num_neighbors}&hybrid_weight={hybrid_weight}&embeddings={embeddings}&reranker_type={reranker_type}&model={model}"


# App Design
st.info(
    """
    # 🤖 Q&A Chatbot Evaluation Tool

    **This is an evaluation tool for question-answering chatbots.** 💡

    - 📝 Create your custom ChatBot for your specific documents.
    - 🕵️ Evaluate the RAG pipeline for its quality in retrieval and question answering.
    - 🧩 Test with different settings to build your perfect ChatBot assistant.

    ## How It Works:
    - **Document Selection**: Choose to upload your own documents or use the pre-loaded Bank Customer Website case. 📁
    - **Ground Truth Options**: Submit your ground truth for detailed and accurate results, or let the app generate it for convenience. 🎯
    - **Compare Different Experiments**: Analyze and compare the results of various experiments to find the best RAG setting. 🔍
    - **Launch Your Custom ChatBot**: Find the ideal RAG combination, launch your custom ChatBot and interact with it live! ✅

    With this tool, you can **experiment with different configurations** and decide what best suits your data to create your perfect ChatBot! 🚀
"""
)

# ChatBot parameters
with st.sidebar.form("user_input"):
    oai_api_key = st.text_input("`OpenAI API Key:`", type="password").strip()

    chunk_size = st.select_slider(
        "`Choose chunk size (in tokens)`", options=[256, 512, 768, 1024], value=256
    )

    chunk_overlap = st.select_slider(
        "`Choose chunk overlap (in tokens)`", options=[0, 20, 50, 100, 200], value=20
    )

    embeddings = st.radio(
        "`Choose embedding model`",
        ("bge-base-en-v1.5", "bge-small-en-v1.5", "bge-large-en-v1.5", "OpenAI"),
        index=0,
    )

    retriever_type = st.radio(
        "`Choose retriever`",
        (
            "Similarity",
            "Hybrid-search",
            "MultiQuery",
            "Similarity + Expansion w/ Generated Answers",
        ),
        index=0,
    )

    hybrid_weight = st.select_slider(
        "`Choose BM25 vs. semantic search weight (0 to 1).\nOnly active for 'Hybrid-search'.`",
        options=[0.00, 0.25, 0.50, 0.75, 1.00],
        value=0.50,
    )

    num_neighbors = st.select_slider(
        "`Choose # chunks to retrieve`", options=[0, 1, 2, 3, 4, 5, 6, 7], value=3
    )

    reranker_type = st.radio(
        "`Choose reranker`",
        ("LongContextReorder", "ms-marco-MiniLM-L-6-v2", "bge-reranker-base", "None"),
        index=0,
    )

    model = st.radio(
        "`Choose ChatBot model`",
        (
            "Mistral-7B-Instruct-v0.1 Small",
            "Mistral-7B-Instruct-v0.1 Med",
            "GPT-4-turbo",
        ),
        index=0,
        key="model",
    )

    model_eval = st.radio(
        "`Choose Evaluator model`",
        (
            "Mistral-7B-Instruct-v0.1 Small",
            "Mistral-7B-Instruct-v0.1 Med",
            "GPT-4-turbo",
        ),
        index=0,
        key="model_eval",
    )

    grade_prompt = st.radio("`Grading style prompt`", ("Binary", "Score"), index=0)

    num_eval_questions = st.slider(
        "`Number of test questions`", min_value=5, max_value=1000, value=10, step=5
    )

    submitted_parameters = st.form_submit_button("Submit parameters")

    if "chatbot_url" not in st.session_state:
        st.session_state.chatbot_url = None

    if submitted_parameters:
        st.session_state.chatbot_url = generate_url(
            oai_api_key,
            chunk_size,
            chunk_overlap,
            embeddings,
            retriever_type,
            hybrid_weight,
            num_neighbors,
            reranker_type,
            model,
        )

    if st.session_state.chatbot_url is not None:
        # Button to launch the chatbot app with Streamlit style
        button_html = f"""<a href="{st.session_state.chatbot_url}" target="_blank">
                            <button style='margin-top: 10px; width: 100%; height: 40px; border: none; border-radius: 20px; background-color: #FF4B4B; color: white;'>
                                Launch Custom ChatBot
                            </button>
                          </a>"""
        st.sidebar.markdown(button_html, unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if "document_selected" not in st.session_state:
    st.session_state.document_selected = False

if "gt_data_ready" not in st.session_state:
    st.session_state.gt_data_ready = False

if "selections_confirmed" not in st.session_state:
    st.session_state.selections_confirmed = False

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = None

if "use_bp_example" not in st.session_state:
    st.session_state.use_bp_example = None

if "gt_option" not in st.session_state:
    st.session_state.gt_option = None

if "detailed_df" not in st.session_state:
    st.session_state.detailed_df = pd.DataFrame()

# Document Upload and Ground Truth Options Section
if not st.session_state.selections_confirmed:
    # Document Upload Section
    st.subheader("Document Upload")
    st.session_state.uploaded_docs = st.file_uploader(
        "Upload documents (.pdf or .txt):",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    st.session_state.use_bp_example = st.checkbox(
        "Use Banco de Portugal - Bank Customer Website"
    )

    # Ground Truth Options Section
    st.subheader("Ground Truth Options")
    gt_options = [
        "Generate Ground Truth from Documents",
        "Upload Own Ground Truth JSON Document",
    ]
    if st.session_state.use_bp_example:
        gt_options.append("Use pre-loaded Bank Customer Website Ground Truth")
    st.session_state.gt_option = st.radio(
        "Choose ground truth data handling:", gt_options
    )

    # Caption with pros and cons of each option
    st.caption(
        """
    📋 Your Ground Truth must be a JSON file containing a list of dictionaries with the following keys:
    - question (your question, a string)
    - answer (your reference answer, a string)
    - sources (chunk IDs of the chunks used to answer the question, a list of integers).

    ⬆️ By uploading your Ground Truth, you can access all retrieval metrics and get more trustworthy results.

    🔗 If you upload your Ground Truth, ensure that the chunking used corresponds to the parameters chosen on the sidebar.
    """
    )

    if st.session_state.gt_option == "Upload Own Ground Truth JSON Document":
        uploaded_gt = st.file_uploader("Upload Ground Truth JSON file:", type="json")

    submit_selections = st.button("Submit Selections")
    if submit_selections:
        st.session_state.document_selected = (
            st.session_state.uploaded_docs is not None
            or st.session_state.use_bp_example
        )
        st.session_state.gt_data_ready = (
            st.session_state.gt_option != "Upload Own Ground Truth JSON Document"
            or uploaded_gt is not None
        )
        st.session_state.selections_confirmed = (
            st.session_state.document_selected and st.session_state.gt_data_ready
        )

# Run Experiment Button
if st.session_state.selections_confirmed:
    st.markdown("---")
    run_experiment = st.button("Run Experiment")

    if run_experiment:
        with st.spinner("`Running Experiment...`"):
            if st.session_state.uploaded_docs:
                combined_text = []
                fnames = []
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
                        st.warning(
                            "Unsupported file type for file: {}".format(file.name)
                        )
                text = " ".join(combined_text)
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", " ", ""],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                texts = text_splitter.split_text(text)
                chunk_ids = [{"chunk_id": i} for i in range(len(texts))]
                chunks = text_splitter.create_documents(texts, chunk_ids)

            elif st.session_state.use_bp_example:
                documents = get_documents()
                chunks = get_chunks(documents, chunk_size, chunk_overlap)

            os.environ["OPENAI_API_KEY"] = oai_api_key
            llm = get_llm(model)
            retriever = get_retriever(
                chunks, retriever_type, embeddings, num_neighbors, llm, hybrid_weight
            )
            qa_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

            if st.session_state.gt_option == "Upload Own Ground Truth JSON Document":
                eval_set = json.load(uploaded_gt)

            elif (
                st.session_state.gt_option
                == "Use pre-loaded Bank Customer Website Ground Truth"
            ):
                gt_path = GROUND_TRUTH_PATH
                with open(gt_path, "r") as json_file:
                    eval_set = json.load(json_file)

            elif st.session_state.gt_option == "Generate Ground Truth from Documents":
                eval_set = generate_eval(chunks, llm, num_eval_questions)

            graded_answers, retrieval_metrics, latency, predictions = run_evaluation(
                qa_chain, retriever, eval_set, model_eval, grade_prompt, retriever_type
            )

        tab1, tab2, tab3 = st.tabs(["Results", "Question Scoring", "Ground Truth"])

        # Assemble outputs
        metrics_df = pd.DataFrame(retrieval_metrics)
        d = pd.DataFrame(predictions)

        # Get the numerical score and the full reasoning
        d["answer score"] = [
            int(re.search(r"\d+", g["results"]).group())
            if re.search(r"\d+", g["results"])
            else 0
            for g in graded_answers
        ]
        d["reasoning"] = [
            re.search(r"Reasoning: (.+)", g["results"], flags=re.DOTALL)
            .group(1)
            .strip()
            if re.search(r"Reasoning: (.+)", g["results"], flags=re.DOTALL)
            else ""
            for g in graded_answers
        ]

        d["latency"] = latency

        # Merge questions with respective metrics
        metrics_df = metrics_df.drop(columns=["question"])
        detailed_df = pd.concat([d, metrics_df], axis=1)
        st.session_state.detailed_df = detailed_df

        mean_latency = d["latency"].mean()
        aggregate_metrics = metrics_df.mean()
        answer_score = d["answer score"].mean()

        # Show less metrics (values of '-') if st.session_state.gt_option == 'Generate Ground Truth from Documents'
        new_row = {
            "answer model": model,
            "retriever": retriever_type,
            "embedding": embeddings,
            "chunk size": chunk_size,
            "chunk overlap": chunk_overlap,
            "k docs retrieved": num_neighbors,
            "reranker": reranker_type,
            "evaluation model": model_eval,
            "evaluation questions": len(eval_set),
            "Answer score": answer_score,
            "Hit Rate": aggregate_metrics["Hit Rate"],
            "MRR": aggregate_metrics["MRR"],
            "NDCG": aggregate_metrics["NDCG"],
            "Precision": aggregate_metrics.get("precision", "-"),
            "Recall": aggregate_metrics.get("recall", "-"),
            "AP": aggregate_metrics.get("AP", "-"),
            "Latency": mean_latency,
        }

        # Append the new experiment data to the existing DataFrame
        new_row_df = pd.DataFrame([new_row])
        summary = pd.concat([summary, new_row_df], ignore_index=True)
        st.session_state.existing_df = summary

        # Define columns for 'Parameters' and 'Results'
        parameters_columns = [
            "answer model",
            "retriever",
            "embedding",
            "chunk size",
            "chunk overlap",
            "k docs retrieved",
            "reranker",
            "evaluation model",
            "evaluation questions",
        ]
        results_columns = [
            col for col in summary.columns if col not in parameters_columns
        ]

        # Tab 1: Aggregate Results of All Experiments
        with tab1:
            st.subheader("`Parameters of All Experiments`")
            parameters_df_all = summary[parameters_columns]
            st.dataframe(parameters_df_all, use_container_width=True)

            st.subheader("`Aggregate Results of All Experiments`")
            # Conditionally display results columns
            if st.session_state.gt_option == "Generate Ground Truth from Documents":
                # Exclude 'Precision', 'Recall', 'AP' from the display
                results_columns_filtered = [
                    col
                    for col in results_columns
                    if col not in ["Precision", "Recall", "AP"]
                ]
            else:
                results_columns_filtered = results_columns

            results_df_all = summary[results_columns_filtered]
            st.dataframe(results_df_all, use_container_width=True)

            # Dataframe for visualization
            show = summary.reset_index()
            show.columns = [
                "expt number",
                "answer model",
                "retriever",
                "embedding",
                "chunk size",
                "chunk overlap",
                "k docs retrieved",
                "reranker",
                "evaluation model",
                "evaluation questions",
                "Answer score",
                "Precision",
                "Recall",
                "Hit Rate",
                "MRR",
                "AP",
                "NDCG",
                "Latency",
            ]
            show["expt number"] = show["expt number"].apply(
                lambda x: "Expt #: " + str(x + 1)
            )

            # Determine the scale based on the value of grade_prompt
            if grade_prompt == "Binary":
                scale_domain = [0, 1]
                y_title = "Binary Score"
            elif grade_prompt == "Score":
                scale_domain = [0, 4]
                y_title = "Score"

            # Set dimensions
            chart_width = 400
            chart_height = chart_width

            c = (
                alt.Chart(show, width=chart_width, height=chart_height)
                .mark_circle()
                .encode(
                    x=alt.X("NDCG", scale=alt.Scale(domain=[0, 1]), title="NDCG"),
                    y=alt.Y(
                        "Answer score",
                        scale=alt.Scale(domain=scale_domain),
                        title=y_title,
                    ),
                    size="Latency",
                    color="expt number",
                    tooltip=["expt number", "NDCG", "Answer score", "Latency"],
                )
            )

            st.altair_chart(c, use_container_width=True)

        # Tab 2: Detailed Results of the Last Experiment
        with tab2:
            st.subheader("`Parameters of Last Experiment`")
            # Parameters of the last experiment
            parameters_df_last = summary.iloc[[-1]][parameters_columns]
            st.dataframe(parameters_df_last, use_container_width=True)

            st.subheader("`Question by Question Detail of Last Experiment`")
            # Modify metrics_df based on st.session_state.gt_option == 'Generate Ground Truth from Documents'
            if st.session_state.gt_option == "Generate Ground Truth from Documents":
                detailed_df_filtered = st.session_state.detailed_df.drop(
                    columns=["precision", "recall", "AP"], errors="ignore"
                )
            else:
                detailed_df_filtered = st.session_state.detailed_df
            st.dataframe(detailed_df_filtered, use_container_width=False)

        # Display in Tab 3: Ground Truth Dataset
        with tab3:
            st.subheader("Ground Truth Data")
            st.dataframe(eval_set)

            # Download button for ground truth data
            json_content = json.dumps(eval_set)
            st.download_button(
                label="Download Ground Truth JSON",
                data=json_content,
                file_name="ground_truth.json",
                mime="application/json",
            )
