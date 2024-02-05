import inspect
from typing import Any, Dict, List, Optional

import torch
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

########################
# MultiQuery Retriever #
########################


# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


output_parser = LineListOutputParser()

MULTIQUERY_TEMPLATE = """<s>[INST] You are an AI language model assistant.
Your task is to generate 3 different search queries that aim to answer the user question from multiple perspectives.
The user questions are focused on Banking, Finance, and related disciplines.
Each query MUST tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.
Provide these alternative questions separated by newlines. [/INST]

Original question: {question}
"""

MULTIQUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=MULTIQUERY_TEMPLATE,
)


####################################
# Expansion with Answer Generation #
####################################

QUERY_EXPANSION_TEMPLATE = """<s>[INST]
Offer a concise and factual answer to the question. The response should be brief and without conversational tone, empathy, greetings, or personal comments.
[/INST]

QUESTION: {new_question}

Possible Answer:

"""

QUERY_EXPANSION_PROMPT = PromptTemplate(
    input_variables=["new_question"],
    template=QUERY_EXPANSION_TEMPLATE,
)


# Retriever that processes the initial query through an LLM Chain
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


#######################
# Follow-up Question #
#######################

QUESTION_GENERATOR_TEMPLATE = """<s>[INST]
Use the chat history to clarify and simplify the follow-up question, ensuring it remains true to the original intent without adding extra details.
[/INST]

Chat History: {chat_history}

Follow-up Question: {question}

Refined Standalone Question:

"""

QUESTION_GENERATOR_PROMPT = PromptTemplate(
    template=QUESTION_GENERATOR_TEMPLATE,
    input_variables=["chat_history", "question"],
)

#############################
# ChatBot Answer Generation #
#############################

ANSWER_TEMPLATE_BDP = """<s>[INST]
Role:
-Act as an AI chat assistant that has been trained exclusively to answer questions about the Bank Customer Website.
-Your goal is to respond to the users' questions using ONLY the information present on the provided [CONTEXT] below.

Rules:
-Make a step by step reasoning about the question.
-The response should be concise and clear.
-Don't mention that you have access to [CONTEXT] or a list of facts.
-You should always refuse to answer questions that are not related to [CONTEXT].
-You SHOULD NOT give any advice or recommendation to the user.
-If the question is out of the [CONTEXT] domain you should briefly and politely respond to the user that you don't have the information to provide the answer.
[/INST]

[CONTEXT]:
{context}

[QUESTION]:
{question}

Helpful Answer:

"""

ANSWER_PROMPT_BDP = PromptTemplate(
    template=ANSWER_TEMPLATE_BDP, input_variables=["context", "question"]
)

ANSWER_TEMPLATE_DEFAULT = """<s>[INST] Use the following pieces of context to answer the question at the end.
If the context doesn't provide enough information to answer the question, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible. [/INST]

Context:
{context}

Question:
{question}

Helpful Answer:

"""

ANSWER_PROMPT_DEFAULT = PromptTemplate(
    template=ANSWER_TEMPLATE_DEFAULT, input_variables=["context", "question"]
)

###################################################
# Custom Chain with follow-up question generation #
###################################################


def reranking(reranker_type, retrieved_docs, question):
    if reranker_type == "LongContextReorder":
        reranker = LongContextReorder()
        reranked_docs = reranker.transform_documents(retrieved_docs)

    elif reranker_type == "ms-marco-MiniLM-L-6-v2":
        cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device=device
        )
        # Get scores for each document
        scores = cross_encoder.predict(
            [[question, doc.page_content] for doc in retrieved_docs]
        )
        # Sort the documents by their scores
        reranked_docs = [
            doc
            for doc, score in sorted(
                zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True
            )
        ]

    elif reranker_type == "bge-reranker-base":
        cross_encoder = CrossEncoder("BAAI/bge-reranker-base", device=device)
        # Get scores for each document
        scores = cross_encoder.predict(
            [[question, doc.page_content] for doc in retrieved_docs]
        )
        # Sort the documents by their scores
        reranked_docs = [
            doc
            for doc, score in sorted(
                zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True
            )
        ]

    elif reranker_type == "None":
        reranked_docs = retrieved_docs

    return reranked_docs


class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
    reranker_type: Optional[str] = Field(
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
                new_question, inputs, self.reranker_type, run_manager=_run_manager
            )
        else:
            docs = self._get_docs(new_question, inputs, self.reranker_type)  # type: ignore[call-arg]
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
        reranker_type: str,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        retrieved_docs = self.retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        docs = reranking(self.reranker_type, retrieved_docs, question)
        return self._reduce_tokens_below_limit(docs)
