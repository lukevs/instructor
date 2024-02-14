from openai import OpenAI
from instructor.client.chat import InstructorOpenAIChat

from instructor.function_calls import Mode


class InstructorOpenAI(OpenAI):
    chat: InstructorOpenAIChat

    def __init__(self, openai_client: OpenAI, mode=Mode.FUNCTIONS) -> None:
        self.__dict__.update(openai_client.__dict__)
        self.chat = InstructorOpenAIChat(openai_client, mode)