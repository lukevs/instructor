from json import JSONDecodeError
import json
import logging
from typing import Callable, Optional, Type, TypeVar, overload

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError
from instructor.dsl.multitask import MultiTaskBase
from instructor.dsl.partial import PartialBase

from instructor.function_calls import Mode

logger = logging.getLogger("instructor")

T_RetryModel = TypeVar("T_RetryModel", bound=BaseModel)
T_RetryFuncReturn = TypeVar("T_RetryFuncReturn")

@overload
def retry_sync(
    func: Callable[..., T_RetryFuncReturn],
    response_model: Type[T_RetryModel],
    validation_context: dict,
    args,
    kwargs,
    max_retries: int = 1,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_RetryModel:
    ...


@overload
def retry_sync(
    func: Callable[..., T_RetryFuncReturn],
    response_model: None,
    validation_context: dict,
    args,
    kwargs,
    max_retries: int = 1,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_RetryFuncReturn:
    ...


def retry_sync(
    func: Callable[..., T_RetryFuncReturn],
    response_model: Type[T_RetryModel] | None,
    validation_context: dict,
    args,
    kwargs,
    max_retries: int = 1,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_RetryFuncReturn | T_RetryModel:
    retries = 0
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    print(args)
    print(kwargs)

    while retries <= max_retries:
        # Excepts ValidationError, and JSONDecodeError
        try:
            response = func(*args, **kwargs)
            stream = kwargs.get("stream", False)

            if isinstance(response, ChatCompletion) and response.usage is not None:
                total_usage.completion_tokens += response.usage.completion_tokens or 0
                total_usage.prompt_tokens += response.usage.prompt_tokens or 0
                total_usage.total_tokens += response.usage.total_tokens or 0
                response.usage = (
                    total_usage  # Replace each response usage with the total usage
                )
            return process_response(
                response,
                response_model=response_model,
                stream=stream,
                validation_context=validation_context,
                strict=strict,
                mode=mode,
            )
        except (ValidationError, JSONDecodeError) as e:
            logger.exception(f"Retrying, exception: {e}")
            logger.debug(f"Error response: {response}")
            kwargs["messages"].append(dump_message(response.choices[0].message))
            if mode == Mode.TOOLS:
                kwargs["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": response.choices[0].message.tool_calls[0].id,
                        "name": response.choices[0].message.tool_calls[0].function.name,
                        "content": "failure",
                    }
                )
            kwargs["messages"].append(
                {
                    "role": "user",
                    "content": f"Recall the function correctly, fix the errors and exceptions found\n{e}",
                }
            )
            if mode == Mode.MD_JSON:
                kwargs["messages"].append(
                    {
                        "role": "assistant",
                        "content": "```json",
                    },
                )
            retries += 1
            if retries > max_retries:
                logger.warning(f"Max retries reached, exception: {e}")
                raise e
            

T_ProcessResponse = TypeVar("T_ProcessResponse")
T_ProcessResponseModel = TypeVar("T_ProcessResponseModel", bound=BaseModel)

@overload
def process_response(
    response: T_ProcessResponse,
    *,
    response_model: Type[T_ProcessResponseModel],
    stream: bool,
    validation_context: dict,
    strict=None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_ProcessResponseModel:
    ...


@overload
def process_response(
    response: T_ProcessResponse,
    *,
    response_model: None,
    stream: bool,
    validation_context: None = None,
    strict=None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_ProcessResponse:
    ...


def process_response(
    response: T_ProcessResponse,
    *,
    response_model: Type[T_ProcessResponseModel] | None,
    stream: bool,
    validation_context: dict = None,
    strict=None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_ProcessResponse | T_ProcessResponseModel:
    """Processes a OpenAI response with the response model, if available.

    Args:
        response (T): The response from OpenAI's API
        response_model (Type[T_Model]): The response model to use for parsing the response
        stream (bool): Whether the response is a stream
        validation_context (dict, optional): The validation context to use for validating the response. Defaults to None.
        strict (_type_, optional): Whether to use strict json parsing. Defaults to None.
        mode (Mode, optional): The openai completion mode. Defaults to Mode.FUNCTIONS.

    Returns:
        Union[T_Model, T]: The parsed response, if a response model is available, otherwise the response as is from the SDK
    """
    if response_model is not None:
        is_model_multitask = issubclass(response_model, MultiTaskBase)
        is_model_partial = issubclass(response_model, PartialBase)
        model = response_model.from_response(
            response,
            validation_context=validation_context,
            strict=strict,
            mode=mode,
            stream_multitask=stream and is_model_multitask,
            stream_partial=stream and is_model_partial,
        )
        if not stream:
            model._raw_response = response
            if is_model_multitask:
                return model.tasks
        return model

    return response


def dump_message(message: ChatCompletionMessage) -> ChatCompletionMessageParam:
    """Dumps a message to a dict, to be returned to the OpenAI API.
    Workaround for an issue with the OpenAI API, where the `tool_calls` field isn't allowed to be present in requests
    if it isn't used.
    """

    ret: ChatCompletionMessageParam = {
        "role": message.role,
        "content": message.content or "",
    }

    if hasattr(message, "tool_calls") and message.tool_calls is not None:
        ret["tool_calls"] = message.model_dump()["tool_calls"]
        ret["content"] += json.dumps(message.model_dump()["tool_calls"], indent=2)

    if hasattr(message, "function_call") and message.function_call is not None:
        ret["content"] += json.dumps(message.model_dump()["function_call"], indent=2)

    return ret
