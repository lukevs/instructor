import logging
from json import JSONDecodeError
from typing import (
    Callable,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
    Union,
    overload,
)

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError
from instructor.client.client import InstructorOpenAI

from instructor.dsl.multitask import MultiTaskBase
from instructor.dsl.partial import PartialBase

from .function_calls import Mode

logger = logging.getLogger("instructor")


T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: Type[T_Model],
    stream: bool = False,
    validation_context: dict = None,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T:
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

    if response_model is not None:
        is_model_multitask = issubclass(response_model, MultiTaskBase)
        is_model_partial = issubclass(response_model, PartialBase)
        model = await response_model.from_response_async(
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


async def retry_async(
    func: Callable[T_ParamSpec, T_Retval],
    response_model: Type[T],
    validation_context,
    args,
    kwargs,
    max_retries,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
):
    retries = 0
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    while retries <= max_retries:
        try:
            response: ChatCompletion = await func(*args, **kwargs)
            stream = kwargs.get("stream", False)
            if isinstance(response, ChatCompletion) and response.usage is not None:
                total_usage.completion_tokens += response.usage.completion_tokens or 0
                total_usage.prompt_tokens += response.usage.prompt_tokens or 0
                total_usage.total_tokens += response.usage.total_tokens or 0
                response.usage = (
                    total_usage  # Replace each response usage with the total usage
                )
            return await process_response_async(
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
            kwargs["messages"].append(dump_message(response.choices[0].message))  # type: ignore
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
            retries += 1
            if retries > max_retries:
                raise e


@overload
def patch(
    client: OpenAI,
    mode: Mode = Mode.FUNCTIONS,
) -> InstructorOpenAI:
    ...


@overload
def patch(
    client: AsyncOpenAI,
    mode: Mode = Mode.FUNCTIONS,
) -> AsyncOpenAI:
    ...


def patch(
    client: Union[OpenAI, AsyncOpenAI],
    mode: Mode = Mode.FUNCTIONS,
) -> Union[InstructorOpenAI, AsyncOpenAI]:
    """
    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """

    logger.debug(f"Patching `client.chat.completions.create` with {mode=}")

    if isinstance(client, AsyncOpenAI):
        # return InstructorAsyncOpenAI(client, mode=mode)
        # TODO
        return client
    else:
        return InstructorOpenAI(client, mode=mode)


def apatch(client: AsyncOpenAI, mode: Mode = Mode.FUNCTIONS):
    """
    No longer necessary, use `patch` instead.

    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """
    import warnings

    warnings.warn(
        "apatch is deprecated, use patch instead", DeprecationWarning, stacklevel=2
    )
    return patch(client, mode=mode)
