import inspect
from json import JSONDecodeError
import json
import logging
from typing import AsyncGenerator, Callable, Optional, Type, TypeVar, overload

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, RetryError, Retrying, stop_after_attempt
from instructor.dsl.iterable import IterableBase
from instructor.dsl.parallel import ParallelBase
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
    max_retries: int | Retrying = 1,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_RetryFuncReturn | T_RetryModel:
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    # If max_retries is int, then create a Retrying object
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}")
        max_retries: Retrying = Retrying(
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
    if not isinstance(max_retries, (Retrying, AsyncRetrying)):
        raise ValueError("max_retries must be an int or a `tenacity.Retrying` object")

    try:
        for attempt in max_retries:
            with attempt:
                try:
                    response = func(*args, **kwargs)
                    stream = kwargs.get("stream", False)
                    if (
                        isinstance(response, ChatCompletion)
                        and response.usage is not None
                    ):
                        total_usage.completion_tokens += (
                            response.usage.completion_tokens or 0
                        )
                        total_usage.prompt_tokens += response.usage.prompt_tokens or 0
                        total_usage.total_tokens += response.usage.total_tokens or 0
                        response.usage = total_usage  # Replace each response usage with the total usage
                    return process_response(
                        response,
                        response_model=response_model,
                        stream=stream,
                        validation_context=validation_context,
                        strict=strict,
                        mode=mode,
                    )
                except (ValidationError, JSONDecodeError) as e:
                    logger.debug(f"Error response: {response}")
                    kwargs["messages"].append(_dump_message(response.choices[0].message))
                    # ! How do we handle this for parallel tools in the future?
                    if mode == Mode.TOOLS:
                        kwargs["messages"].append(
                            {
                                "role": "tool",
                                "tool_call_id": response.choices[0]
                                .message.tool_calls[0]
                                .id,
                                "name": response.choices[0]
                                .message.tool_calls[0]
                                .function.name,
                                "content": f"Recall the function correctly, fix the errors and exceptions found\n{e}",
                            }
                        )
                    else:
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
                    raise e
    except RetryError as e:
        logger.exception(f"Failed after retries: {e.last_attempt.exception}")
        raise e.last_attempt.exception from e


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
    response_model: Type[T_ProcessResponseModel],
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
    if response_model is None:
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = response_model.from_streaming_response(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        return model

    model._raw_response = response
    return model


T_AsyncRetryModel = TypeVar("T_AsyncRetryModel", bound=BaseModel)
T_AsyncRetryFuncReturn = TypeVar("T_AsyncRetryFuncReturn")

async def retry_async(
    func: Callable[..., T_AsyncRetryFuncReturn],
    response_model: Type[T_AsyncRetryModel] | None,
    validation_context,
    args,
    kwargs,
    max_retries: int | AsyncRetrying = 1,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_AsyncRetryFuncReturn | T_AsyncRetryModel:
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    # If max_retries is int, then create a AsyncRetrying object
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}")
        max_retries = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
    if not isinstance(max_retries, (AsyncRetrying, Retrying)):
        raise ValueError(
            "max_retries must be an `int` or a `tenacity.AsyncRetrying` object"
        )

    try:
        async for attempt in max_retries:
            logger.debug(f"Retrying, attempt: {attempt}")
            with attempt:
                try:
                    response: ChatCompletion = await func(*args, **kwargs)
                    stream = kwargs.get("stream", False)
                    if (
                        isinstance(response, ChatCompletion)
                        and response.usage is not None
                    ):
                        total_usage.completion_tokens += (
                            response.usage.completion_tokens or 0
                        )
                        total_usage.prompt_tokens += response.usage.prompt_tokens or 0
                        total_usage.total_tokens += response.usage.total_tokens or 0
                        response.usage = total_usage  # Replace each response usage with the total usage
                    return await process_response_async(
                        response,
                        response_model=response_model,
                        stream=stream,
                        validation_context=validation_context,
                        strict=strict,
                        mode=mode,
                    )
                except (ValidationError, JSONDecodeError) as e:
                    logger.debug(f"Error response: {response}")
                    kwargs["messages"].append(_dump_message(response.choices[0].message))  # type: ignore
                    if mode == Mode.TOOLS:
                        kwargs["messages"].append(
                            {
                                "role": "tool",
                                "tool_call_id": response.choices[0]
                                .message.tool_calls[0]
                                .id,
                                "name": response.choices[0]
                                .message.tool_calls[0]
                                .function.name,
                                "content": "Exceptions found\n{e}\nRecall the function correctly.",
                            }
                        )

                    kwargs["messages"].append(
                        {
                            "role": "user",
                            "content": f"Recall the function correctly, fix the errors, exceptions found\n{e}",
                        }
                    )
                    if mode == Mode.MD_JSON:
                        kwargs["messages"].append(
                            {
                                "role": "assistant",
                                "content": "```json",
                            },
                        )
                    raise e
    except RetryError as e:
        logger.exception(f"Failed after retries: {e.last_attempt.exception}")
        raise e.last_attempt.exception from e



T_AsyncProcessResponse = TypeVar("T_AsyncProcessResponse")
T_AsyncProcessResponseModel = TypeVar("T_AsyncProcessResponseModel", bound=BaseModel)


@overload
async def process_response_async(
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
def process_response_async(
    response: T_AsyncProcessResponse,
    *,
    response_model: None,
    stream: bool = False,
    validation_context: dict = None,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_AsyncProcessResponse:
    ...


async def process_response_async(
    response: T_AsyncProcessResponse,
    *,
    response_model: Type[T_AsyncProcessResponseModel] | None,
    stream: bool = False,
    validation_context: dict = None,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_AsyncProcessResponse | T_AsyncProcessResponseModel | AsyncGenerator[T_AsyncProcessResponseModel, None]:
    """Processes a OpenAI response with the response model, if available.
    It can use `validation_context` and `strict` to validate the response
    via the pydantic model

    Args:
        response (ChatCompletion): The response from OpenAI's API
        response_model (BaseModel): The response model to use for parsing the response
        stream (bool): Whether the response is a stream
        validation_context (dict, optional): The validation context to use for validating the response. Defaults to None.
        strict (bool, optional): Whether to use strict json parsing. Defaults to None.
    """

    if response_model is None:
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = await response_model.from_streaming_response_async(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        #! If the response model is a multitask, return the tasks
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        return model

    model._raw_response = response
    return model


def _dump_message(message: ChatCompletionMessage) -> ChatCompletionMessageParam:
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
    if hasattr(message, "function_call") and message.function_call is not None:
        ret["content"] += json.dumps(message.model_dump()["function_call"])
    return ret
