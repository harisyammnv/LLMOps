RAQA_PROMPT_TEMPLATE = """
Use the provided context to answer the user's query. 

You may not answer the user's query unless there is specific context in the following text.

If you do not know the answer, or cannot answer, please respond with "I don't know".

Context:
{context}
"""

USER_PROMPT_TEMPLATE = """
User Query:
{user_query}
"""