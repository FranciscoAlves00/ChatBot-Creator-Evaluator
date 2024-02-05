# ChatBot Evaluation and Creating Tool

## Description
This innovative tool is designed for the evaluation and creation of question-answering chatbots. It provides a user-friendly platform to:

- **📝 Create Your Custom ChatBot** for your specific documents.
- **🕵️ Evaluate the RAG Pipeline** for its quality in retrieval and question answering.
- **🧩 Test With Different Settings** to build your perfect ChatBot assistant.

**How It Works:**
- **Document Selection:** Choose to upload your own documents or use the pre-loaded Bank Customer Website case. 📁
- **Ground Truth Options:** Submit your ground truth for detailed and accurate results, or let the app generate it for convenience. 🎯
- **Compare Different Experiments:** Analyze and compare the results of various experiments to find the best RAG setting. 🔍
- **Launch Your Custom ChatBot:** Find the ideal RAG combination, create your custom ChatBot, and interact with it live! ✅

With this tool, you can experiment with different configurations and decide what best suits your data to create your perfect ChatBot! 🚀

## Installation Instructions

### Prerequisites

Before you begin the installation process, ensure you have Git LFS installed on your machine. Git LFS is required to properly download the Large Language Models (Mistral LLM) used in this project. If you do not have Git LFS installed, follow the instructions on the [Git LFS website](https://git-lfs.github.com/) to set it up on your system.

### Setting Up the Project

1. **Clone the Repository**: Start by cloning this repository to your local machine. Ensure Git LFS is installed to handle the large files correctly:

    ```sh
    git clone https://github.com/FranciscoAlves00/ChatBot-Creator-Evaluator.git
    cd ChatBot-Creator-Evaluator
    ```

2. **Install Dependencies**: Install the required Python packages using the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

    This project relies on specific dependencies managed via a `requirements.txt` file.

### Working with Local Models

To work with local models, it is necessary to install the Mistral Q2_K and Q4_K_M quantized methods. These models offer optimized performance for language understanding tasks. For detailed instructions on installing these quantized models, please consult [Mistral-7B-Instruct-v0.1-GGUF on Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF).

### Optional: GPU Acceleration with LlamaCPP

To achieve significantly improved inference times, users with access to a GPU are strongly advised to install `llama.cpp` using cuBLAS. This enhancement leverages GPU acceleration for faster LLM inference. While this step is optional, enabling it by uncommenting the relevant line in the `requirements.txt` file can substantially boost performance.

For more information, please visit [LlamaCPP Integration Guide](https://python.langchain.com/docs/integrations/llms/llamacpp).

## Usage
To create and interact with your custom chatbot, follow these steps:
1. **Prepare the Environment:** Ensure that `chatbot.py` is executed first to set up the necessary environment for your custom chatbot.
2. **Run the Evaluation Tool:** Execute the `autoeval.py` file to start the AutoEval application, and follow the on-screen instructions.
3. **Experiment Setup:** On the sidebar, select your RAG and testing parameters, choose the documents for submission along with the ground truth base, and initiate the experiment.
4. **Analysis:** Run multiple experiments if necessary, and review the results presented in tables and graphs.
5. **Create Custom ChatBot:** Once satisfied with the setup, select the correct parameters again and click on "Create Custom ChatBot." A new tab will open, allowing you to interact with your ChatBot directly.

## Contact
For any inquiries or collaboration opportunities, feel free to contact Francisco Alves at [francisco.2000.alves@gmail.com](mailto:francisco.2000.alves@gmail.com).
