# Project Context: ArXiv AI Assistant

## The Goal
I want to build a Python web application that allows users to paste a link to an ArXiv research paper (e.g., `https://arxiv.org/abs/2305.10601`) and ask questions about it. The app will act as an interactive AI agent that reads the full paper and answers the user's queries based *only* on the text of that paper.

## Tech Stack
* **Language:** Python
* **LLM Orchestration:** LangChain (`langchain`, `langchain-community`, `langchain-openai`)
* **Document Loader:** `ArxivLoader` from `langchain_community.document_loaders`
* **LLM Provider:** OpenAI (`gpt-5.5`) or Grok (via `langchain-openai` pointing to the xAI base URL and using `grok-4.3`)
* **Frontend UI:** Gradio (`gradio`)

## Step-by-Step Requirements

### 1. URL Parsing & Validation
* Create a robust function that takes user input (a URL) and uses regex to extract just the ArXiv ID (e.g., extracting `2305.10601` from `https://arxiv.org/abs/2305.10601` or `https://arxiv.org/pdf/2305.10601.pdf`).
* Provide clear error messages in the UI if the URL is invalid.

### 2. Document Loading
* Use LangChain's `ArxivLoader` with the extracted ID to fetch the paper.
* Set `load_max_docs=1` to ensure we only get the specific paper.
* Extract the `page_content` and metadata (like the paper's 'Title' and 'Authors') so it can be displayed as context in the interface once successfully parsed.

### 3. LLM Integration, Memory & System Prompt
* Initialize the LLM using `ChatOpenAI`. 
  * Make it flexible so it can use **OpenAI** (defaulting to the flagship `gpt-5.5` model) OR **Grok** by adjusting the `openai_api_base` to xAI's endpoint (`https://api.x.ai/v1`) and passing the `grok-4.3` model name.
* **Crucial System Prompt Design:** You must design a system prompt that forces the LLM to act as a captivating science communicator. When summarizing or explaining the paper, the LLM MUST:
  * Extract and cover these key aspects: Motivation, Context of the Problem, Proposed Solution, How it Works, Comparison to Existing Methods, and Future Work.
  * Use **layman's terms** only. The explanation must be easy to understand and explain to a non-expert, avoiding heavy academic jargon.
  * Use a captivating, interesting "hook" style to make the reader want to go read the actual paper.
  * Structure the output as a **"bite-sized carousel"**. Since this is text, format it using distinct, visually separated Markdown blocks (e.g., using `---` horizontal rules, emojis, and bold headers like `### 🎠 Slide 1: The Problem`) so it feels like the user is swiping through an engaging social media carousel.
* Implement a stateful conversation history using Gradio's state management or LangChain's memory so the user can have a multi-turn chat about the document.

### 4. Gradio User Interface
* Build a clean, modern interface using `gr.Blocks()`.
* **Sidebar / Configuration Panel:**
  * A dropdown (`gr.Dropdown`) to choose the provider: **OpenAI (GPT-5.5)** or **Grok (Grok 4.3)**.
  * A field to input the respective API Key (mask the input).
  * A text input for the ArXiv URL.
  * A "Load Paper" button.
  * A text or markdown component to display the loaded paper's Title/Authors upon success.
* **Main Area:**
  * A conversational chat interface using `gr.Chatbot`.
  * A text input for the user's message and a submit button.
  * Ensure the chatbot streams responses for a smooth user experience.

## Code Guidelines & Expected Output
Please act as an expert Python engineer and provide the following:
1. **`requirements.txt`**: The exact pip packages needed to run this.
2. **`app.py`**: The complete, runnable Gradio application code.
3. Add inline comments explaining the LangChain components, state management, and streaming implementation.
4. Ensure the code handles errors gracefully (e.g., API key missing, document failed to load, API rate limits).
