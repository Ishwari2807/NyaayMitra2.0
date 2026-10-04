from langchain_ollama import ChatOllama


class LangChainLLMClient:

    def __init__(
        self,
        model: str = "llama3.2:1b",
    ):
        self.llm = ChatOllama(
            model=model,
            temperature=0.1,
            format="json",
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        response = self.llm.invoke(
            [
                (
                    "system",
                    system_prompt,
                ),
                (
                    "user",
                    user_prompt,
                ),
            ]
        )

        return response.content