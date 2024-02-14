from typing import Iterable, Tuple, Type, TypeVar, get_args, get_origin
from pydantic import BaseModel
from instructor.dsl.multitask import MultiTask, MultiTaskBase
from instructor.dsl.partial import PartialBase

from instructor.function_calls import Mode, OpenAISchema, openai_schema


T_ResponseModel = TypeVar("T_ResponseModel", bound=BaseModel)


def handle_response_model(
    response_model: T_ResponseModel, mode: Mode = Mode.TOOLS, **kwargs
) -> Tuple[Type[OpenAISchema], dict]:
    """Prepare the response model type hint, and returns the response_model
    along with the new modified kwargs needed to be able to use the response_model
    parameter with the patch function.

    Args:
        response_model (T): The response model to use for parsing the response
        mode (Mode, optional): The openai completion mode. Defaults to Mode.TOOLS.

    Raises:
        NotImplementedError: When using stream=True with a non-iterable response_model
        ValueError: When using an invalid patch mode

    Returns:
        Union[Type[OpenAISchema], dict]: The response model to use for parsing the response
    """

    new_kwargs = kwargs.copy()

    if response_model is not None:
        if get_origin(response_model) is Iterable:
            iterable_element_class = get_args(response_model)[0]
            response_model = MultiTask(iterable_element_class)

        if not issubclass(response_model, OpenAISchema):
            response_model = openai_schema(response_model)  # type: ignore

        if new_kwargs.get("stream", False) and not issubclass(
            response_model, (MultiTaskBase, PartialBase)
        ):
            raise NotImplementedError(
                "stream=True is not supported when using response_model parameter for non-iterables"
            )

        if mode == Mode.FUNCTIONS:
            new_kwargs["functions"] = [response_model.openai_schema]  # type: ignore
            new_kwargs["function_call"] = {"name": response_model.openai_schema["name"]}  # type: ignore

        elif mode == Mode.TOOLS:
            new_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": response_model.openai_schema,
                }
            ]
            new_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": response_model.openai_schema["name"]},
            }

        elif mode in {Mode.JSON, Mode.MD_JSON, Mode.JSON_SCHEMA}:
            # If its a JSON Mode we need to massage the prompt a bit
            # in order to get the response we want in a json format
            message = f"""
                As a genius expert, your task is to understand the content and provide 
                the parsed objects in json that match the following json_schema:\n
                {response_model.model_json_schema()['properties']}
                """
            # Check for nested models
            if "$defs" in response_model.model_json_schema():
                message += f"\nHere are some more definitions to adhere too:\n{response_model.model_json_schema()['$defs']}"

            if mode == Mode.JSON:
                new_kwargs["response_format"] = {"type": "json_object"}

            elif mode == Mode.JSON_SCHEMA:
                new_kwargs["response_format"] = {
                    "type": "json_object",
                    "schema": response_model.model_json_schema(),
                }

            elif mode == Mode.MD_JSON:
                new_kwargs["messages"].append(
                    {
                        "role": "assistant",
                        "content": "Here is the perfectly correctly formatted JSON\n```json",
                    },
                )
                new_kwargs["stop"] = "```"
            # check that the first message is a system message
            # if it is not, add a system message to the beginning
            if new_kwargs["messages"][0]["role"] != "system":
                new_kwargs["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": message,
                    },
                )

            # if the first message is a system append the schema to the end
            if new_kwargs["messages"][0]["role"] == "system":
                new_kwargs["messages"][0]["content"] += f"\n\n{message}"
        else:
            raise ValueError(f"Invalid patch mode: {mode}")

    return response_model, new_kwargs
