# LangChain Learning Journey

## Overview

This repository documents my hands-on journey of learning **LangChain, Large Language Models (LLMs), and AI application development using Python**.

The objective is to understand LangChain fundamentals, implement reusable AI workflows, manage conversational interactions, and build practical applications using Streamlit.

My learning approach combines theoretical understanding, small coding exercises, debugging, and end-to-end projects.

## 1. Technologies Used

- **Python** — Application development
- **LangChain** — LLM application framework
- **OpenAI API** — Language model integration
- **Pydantic** — Structured data validation
- **Streamlit** — Interactive web interfaces
- **Python-dotenv** — Environment variable management
- **Git and GitHub** — Version control and project management

## 2. LangChain Concepts Covered

### 2.1 Chat Models

Learned how to interact with language models using LangChain.

Topics covered:
- Initializing chat models using `ChatOpenAI`
- Invoking models using `.invoke()`
- Working with user prompts and model responses
- Understanding model input and output
- Configuring model selection

**Hands-on practice:** Built simple Python programs that send questions to an LLM and display generated responses.

### 2.2 Prompt Templates

Learned how to create reusable, dynamic prompts.

Topics covered:
- `PromptTemplate`
- `ChatPromptTemplate`
- Input variables
- Partial variables
- System messages
- Human messages
- AI messages
- Prompt formatting

**Hands-on practice:** Created prompts for explanations, feedback analysis, sentiment classification, and customer response generation.

### 2.3 Output Parsers

Learned how to convert LLM responses into usable application data.

Topics covered:
- `StrOutputParser`
- `PydanticOutputParser`
- Structured JSON output
- Output formatting instructions
- Handling parsing errors

**Hands-on practice:** Converted LLM responses into strings and validated structured Pydantic objects.

### 2.4 Pydantic and Structured Output

Learned how to define structured output schemas.

Topics covered:
- `BaseModel`
- `Field`
- `Literal`
- `list[str]`
- Field descriptions
- Schema-based output validation

**Hands-on practice:** Designed a customer feedback analysis schema containing:

- Sentiment
- Summary
- Pros
- Cons
- Key themes

### 2.5 LangChain Expression Language (LCEL)

Learned how to combine LangChain components into reusable pipelines.

Topics covered:
- Pipe operator (`|`)
- Runnable interfaces
- Chain composition
- `.invoke()`
- Passing data between chain components

**Hands-on practice:** Connected prompts, language models, and output parsers into working chains.

### 2.6 RunnableSequence

Learned how to execute multiple processing steps sequentially.

Topics covered:
- Sequential execution
- Passing output between components
- Building multi-step workflows

**Hands-on practice:** Created workflows that process inputs through prompts, models, and parsers.

### 2.7 RunnableLambda

Learned how to integrate custom Python functions into LangChain pipelines.

Topics covered:
- Lambda expressions
- Transforming dictionaries
- Formatting model output
- Extracting structured fields
- Preparing data for subsequent processing

**Hands-on practice:** Converted structured analysis results into dictionaries and prepared inputs for downstream chains.

### 2.8 RunnableParallel

Learned how to organize independent operations within a LangChain workflow.

Topics covered:
- Parallel runnable composition
- Dictionary-based outputs
- Combining independent results

**Hands-on practice:** Practised composing independent chain operations.

### 2.9 RunnableBranch

Learned how to implement conditional execution in LangChain.

Topics covered:
- Conditional routing
- Branch selection
- Default branches
- Sentiment-based workflows

**Hands-on practice:** Created a feedback workflow that generates different customer replies depending on whether sentiment is Positive, Negative, or Mixed.

### 2.10 RunnablePassthrough

Learned how to preserve existing inputs while enriching a workflow with additional results.

Topics covered:
- Passing original inputs forward
- `RunnablePassthrough.assign()`
- Adding analysis results to dictionaries
- Maintaining multiple intermediate values

**Hands-on practice:** Built a workflow that preserves customer feedback while adding sentiment analysis and a generated reply.

## 3. Conversation History and Memory

Studied how conversational applications maintain context between user messages.

Topics covered:
- Conversation history
- `MessagesPlaceholder`
- `InMemoryChatMessageHistory`
- `RunnableWithMessageHistory`
- Session identifiers
- Multiple conversation sessions
- Streamlit Session State
- Maintaining user and assistant messages
- Passing previous messages to an LLM

### Key Learning

LLMs do not automatically remember previous interactions.

Applications must manage conversation history and provide relevant previous messages to the model when generating a response.

### Hands-on Practice

Implemented a conversational assistant capable of remembering information shared earlier in the same session.

For example:

**User:** My smartphone is an iPhone 14.

**User:** Which smartphone did I mention?

**AI:** You mentioned an iPhone 14.

**Area for further practice:** Independently implementing conversation history and understanding Streamlit rerun behavior.

## 4. Streamlit Application Development

Learned how to create interactive AI applications with Streamlit.

Topics covered:
- `st.title()`
- `st.write()`
- `st.text_area()`
- `st.selectbox()`
- `st.button()`
- `st.spinner()`
- `st.subheader()`
- `st.markdown()`
- `st.chat_input()`
- `st.chat_message()`
- `st.session_state`

Also practised:
- Conditional UI rendering
- Connecting frontend and backend modules
- Displaying structured output
- Formatting lists as Markdown bullets
- Managing chat interfaces
- Handling empty user inputs

## 5. Project 1 — AI Learning Assistant

Built an interactive learning application using LangChain and Streamlit.

### Features

**Explain Mode**
- Accepts a topic from the user
- Generates an explanation
- Produces structured educational content

**Quiz Mode**
- Generates quiz questions about a selected topic
- Provides an interactive learning experience

**Ask a Question Mode**
- Allows conversational questions
- Uses conversation history to maintain context

### Concepts Applied

- ChatPromptTemplate
- PydanticOutputParser
- Runnable chains
- MessagesPlaceholder
- Conversation history
- Streamlit UI
- Session management

**Outcome:** Completed a working multi-mode educational assistant.

## 6. Project 2 — AI Customer Feedback Assistant

Built a customer feedback processing application using LangChain, Pydantic, and Streamlit.

### Mode 1 — Feedback Analysis

Analyzes customer reviews and generates structured output containing:

- Sentiment: Positive, Negative, or Mixed
- Summary
- Pros
- Cons
- Key themes

**Concepts applied:** Prompt templates, Pydantic schemas, output parsers, and LCEL chains.

### Mode 2 — Customer Reply Generation

Generates a suitable customer response based on feedback sentiment.

Workflow:

1. Receive customer feedback.
2. Analyze the feedback.
3. Extract sentiment and other relevant fields.
4. Select an appropriate response chain.
5. Generate a customer reply.
6. Display the analysis and reply.

**Concepts applied:** RunnablePassthrough, RunnableLambda, RunnableBranch, and structured output.

### Mode 3 — Conversational AI

Implements a conversational customer support assistant.

Features:
- Interactive chat interface
- User and assistant chat messages
- Conversation history
- Context-aware follow-up responses
- Streamlit Session State

**Concepts applied:** ChatPromptTemplate, MessagesPlaceholder, RunnableSequence, and session state management.

### Project Outcome

Successfully developed and manually tested all three functional modes.

The application demonstrates how LangChain components can be combined into a practical multi-feature AI application.

## 7. Debugging and Problem Solving

During development, encountered and worked through several practical issues:

- PromptTemplate configuration errors
- Pydantic parsing failures
- Incorrect dictionary keys
- Conditional branch errors
- Circular imports between Python modules
- Streamlit button and rerun behavior
- Conversation history implementation
- Model availability and deprecation errors
- Formatting structured results for display

These exercises strengthened understanding of debugging, data flow, and integrating backend logic with a frontend.

## 8. Git and GitHub

Learned the fundamentals of Git version control.

Commands practised:

- `git init`
- `git status`
- `git add`
- `git commit`
- `git remote add origin`
- `git branch -M main`
- `git push`

Also learned how to use `.gitignore` to exclude sensitive or unnecessary files, including API keys, environment files, and Python cache directories.

## 9. Current Learning Status

| Topic | Status |
|---|---|
| Chat Models | Completed |
| Prompt Templates | Completed |
| Output Parsers | Completed |
| Pydantic Structured Output | Completed |
| LCEL and Runnable Chains | Completed |
| RunnableSequence | Completed |
| RunnableLambda | Completed |
| RunnableParallel | Completed |
| RunnableBranch | Completed |
| RunnablePassthrough | Completed |
| Conversation History | Implemented — Further Practice Planned |
| Streamlit | Implemented — Further Practice Planned |
| AI Learning Assistant | Completed |
| AI Customer Feedback Assistant | Completed |
| Git and GitHub Basics | Completed |

## 10. Upcoming Learning Goals

The next phase focuses on Retrieval-Augmented Generation (RAG) and Agentic AI development.

Planned topics include:

- Document loaders
- Text splitting and chunking
- Embeddings
- Vector databases
- Similarity search
- Retrievers
- Retrieval-Augmented Generation
- Conversational RAG
- LangGraph
- Human-in-the-loop workflows
- Multi-agent systems
- AI guardrails
- Evaluation and testing
- Deployment

### Next Major Project

**Enterprise Knowledge Base Assistant**

The goal is to build an end-to-end RAG application that can ingest documents, retrieve relevant information, generate grounded answers, and provide source references.

## 11. Long-Term Objective

To become proficient in building practical, production-oriented AI applications by combining:

- Strong Python and LangChain fundamentals
- LLM application architecture
- Retrieval-Augmented Generation
- Agentic workflows
- State and memory management
- Testing and evaluation
- Deployment
- GitHub portfolio development

The ultimate goal is to develop end-to-end projects that demonstrate practical problem-solving skills and can be presented during technical interviews.

---

**Learning philosophy:** Understand the concept, implement it independently, debug problems, and build real applications.