from typing import List

import torch
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel, Field

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
    chain: LLMChain
    vectorstore: VectorStoreRetriever
    search_kwargs: dict = Field(default_factory=dict)

    def get_relevant_documents(self, query: str) -> List[Document]:
        output = self.chain.invoke(query)
        new_query = output["new_question"] + "\n" + output["query"]
        results = self.vectorstore.get_relevant_documents(query=new_query)
        return results


######################
# Evaluate Answering #
######################

binary_answer_template = """<s>[INST] Your task is to evaluate the quality of our generated answer compared to a reference answer for a given query.
Please structure your response as follows:
- Start with "Score:" followed by a numerical score of EITHER 0 or 1. Use 0 if the generated answer is incorrect, and 1 if it is correct.
- On a new line, start with "Reasoning:" and provide your reasoning for the score given. Make sure to explain why the generated answer is correct or incorrect in relation to the query and reference answer.

Your adherence to this response format is crucial for accurate assessment.
[/INST]

QUERY: {query}

GENERATED ANSWER: {result}

REFERENCE ANSWER: {answer}

RESPONSE:

"""

BINARY_ANSWER_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=binary_answer_template
)


score_answer_template = """<s>[INST] Your task is to evaluate the quality of our generated answer compared to a reference answer for a given query.
Please structure your response as follows:
- Start with "Score:" followed by a numerical score between 0 and 4. Use ONLY integer values (0, 1, 2 or 4).
- On a new line, start with "Reasoning:" and provide your reasoning for the score given. Make sure to explain why the generated answer is correct or incorrect in relation to the query and reference answer.

Your adherence to this response format is crucial for accurate assessment.
[/INST]

QUERY: {query}

GENERATED ANSWER: {result}

REFERENCE ANSWER: {answer}

RESPONSE:

"""

SCORE_ANSWER_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=score_answer_template
)

############################
# ChatBot Answering Prompt #
############################

template = """<s>[INST] Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum. Keep the answer as concise as possible. [/INST]

Context:
{context}

Question:
{question}

Helpful Answer:

"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)
