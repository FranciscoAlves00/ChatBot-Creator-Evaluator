from langchain.prompts import PromptTemplate

HYDE_TEMPLATE = """<s>[INST]
Please offer a concise and factual answer to the question. Use three sentences maximum.
[/INST]

QUESTION: {new_question}

Possible Answer:

"""

HYDE_PROMPT = PromptTemplate(
    input_variables=["new_question"],
    template=HYDE_TEMPLATE,
)

QA_CHAIN_TEMPLATE = """<s>[INST] Use the following context to answer the question.
If you're unsure, simply say "I don't have enough information to answer" instead of trying to make up an answer.
Keep your answer concise. Use three sentences maximum. [/INST]

Context:
{context}

Question:
{question}

Helpful Answer:

"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=QA_CHAIN_TEMPLATE
)


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
