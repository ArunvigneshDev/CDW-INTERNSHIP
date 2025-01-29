from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

def code_generator_prompt():
    """
    Generates Prompt template for basic code generation.
    """
    system_msg = '''
                You are an expert code generator. Your task is strictly to generate clean, 
                optimized, and well-commented code for the given task or functionality. 
                Follow these guidelines:
                
                1. Generate only code based on the task provided by the user.
                2. Avoid unnecessary explanations or headers unless requested.
                3. Ensure proper coding style, syntax, and comments for the specified programming language.
                4. Respond with the generated code or a message requesting clarification 
                if the task is ambiguous.
                '''

    user_msg = "Generate a code snippet in {language} for the following task: {task}"

    prompt_template = ChatPromptTemplate([
        ("system", system_msg),
        ("user", user_msg)
    ])

    return prompt_template


def code_generator_rag_prompt():
    """
    Generates a RAG-enabled Prompt template for code generation.
    """
    system_msg = '''
                You are an expert code generator who incorporates relevant 
                external knowledge and context for generating optimized solutions. 
                Follow these guidelines:

                1. Use the retrieved context for generating code that aligns with the task.
                2. Always provide clean, efficient, and commented code solutions in the specified programming language.
                3. If the retrieved context doesn't relate to the task, default to general 
                programming knowledge.
                '''
    
    user_msg = "Generate a code snippet in {language} for the following task: {task}, using the context: {context}"

    prompt_template = ChatPromptTemplate([
        ("system", system_msg),
        ("user", user_msg)
    ])

    return prompt_template