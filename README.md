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

## Getting Started:

To get started with this evaluation tool, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine to get started.
   ```bash
   git clone https://github.com/FranciscoAlves00/rag-chatbot-eval.git
   cd rag-chatbot-eval
   ```
2. **Install Dependencies**: Install the required dependencies using the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**: Start the application using the command below and follow the instructions on your screen.
   ```bash
   streamlit run app.py
   ```

## Contact
For any inquiries or collaboration opportunities, feel free to contact Francisco Alves at [francisco.2000.alves@gmail.com](mailto:francisco.2000.alves@gmail.com).