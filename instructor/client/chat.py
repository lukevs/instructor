from openai.resources import Chat
from instructor.client.completions import InstructorOpenAIChatCompletions

from instructor.function_calls import Mode


class InstructorOpenAIChat(Chat):
    completions: InstructorOpenAIChatCompletions

    def __init__(self, openai_chat: Chat, mode: Mode):
        self.__dict__.update(openai_chat.__dict__)
        self.completions = InstructorOpenAIChatCompletions(
            openai_chat.completions, mode
        )
