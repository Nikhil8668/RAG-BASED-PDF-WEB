

#I have used type ignore comments
#  to avoid type checking issues for libraries that may not have type stubs available.
from openai import OpenAI  # type: ignore 
import os

# Initialize OpenAI client

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def get_openai_client():
    """Return the OpenAI client instance."""
    return client



def ask_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Ask the LLM a question and get a response.

    Args:
        prompt (str): The user prompt.
        model (str): The model name (default: gpt-4o-mini).

    Returns:
        str: The response from the model.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )


    return response.choices[0].message.content.strip()


