from langchain_core.output_parsers import StrOutputParser
import models
import prompts
import vectordb

#### GENERATION ####
def generate_code_chain(task, language):
    """
    Generate code snippet using basic prompt LLM chain.

    Args:
        task - description of the programming task or functionality
        language - programming language for the code generation

    Returns:
        response.content -> str
    """
    llm = models.create_chat_groq_model()

    prompt_template = prompts.code_generator_prompt()

    chain = prompt_template | llm

    response = chain.invoke({
        "task": task,
        "language": language  # Pass the language to the prompt
    })
    return response.content


#### RETRIEVAL and GENERATION ####
def generate_code_rag_chain(task, language, vector):
    """
    Creates a RAG chain for retrieval and code generation.

    Args:
        task - description of the programming task
        language - programming language for the code generation
        vectorstore -> Instance of vector store 

    Returns:
        response -> str
    """
    # Prompt
    prompt = prompts.code_generator_rag_prompt()

    # LLM
    llm = models.create_chat_groq_model()

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectordb.retrieve_from_chroma(task, vectorstore=vector)
    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({
        "context": format_docs(retriever),
        "task": task,
        "language": language  # Pass the language to the prompt
    })

    return response