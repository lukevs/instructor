import inspect
import json
import logging
from collections.abc import Iterable
from json import JSONDecodeError
from typing import (
    Callable,
    Generic,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)

from openai import resources
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError

from instructor.dsl.multitask import MultiTask, MultiTaskBase
from instructor.dsl.partial import PartialBase

from .function_calls import Mode, OpenAISchema, openai_schema

logger = logging.getLogger("instructor")


T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


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


def handle_response_model(
    response_model: T, mode: Mode = Mode.TOOLS, **kwargs
) -> Union[Type[OpenAISchema], dict]:
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

T_ProcessResponse = TypeVar("T_ProcessResponse")
T_ProcessResponseModel = TypeVar("T_ProcessResponseModel", bound=BaseModel)

@overload
def process_response(
    response: T_ProcessResponse,
    *,
    response_model: Type[T_ProcessResponseModel],
    stream: bool,
    validation_context: dict = None,
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
    validation_context: dict = None,
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

T_RetryModel = TypeVar("T_RetryModel")
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


def is_async(func: Callable) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    return inspect.iscoroutinefunction(func) or (
        hasattr(func, "__wrapped__") and inspect.iscoroutinefunction(func.__wrapped__)
    )


OVERRIDE_DOCS = """
Creates a new chat completion for the provided messages and parameters.

See: https://platform.openai.com/docs/api-reference/chat-completions/create

Additional Notes:

Using the `response_model` parameter, you can specify a response model to use for parsing the response from OpenAI's API. If its present, the response will be parsed using the response model, otherwise it will be returned as is. 

If `stream=True` is specified, the response will be parsed using the `from_stream_response` method of the response model, if available, otherwise it will be parsed using the `from_response` method.

If need to obtain the raw response from OpenAI's API, you can access it using the `_raw_response` attribute of the response model. The `_raw_response.usage` attribute is modified to reflect the token usage from the last successful response as well as from any previous unsuccessful attempts.

Parameters:
    response_model (Union[Type[BaseModel], Type[OpenAISchema]]): The response model to use for parsing the response from OpenAI's API, if available (default: None)
    max_retries (int): The maximum number of retries to attempt if the response is not valid (default: 0)
    validation_context (dict): The validation context to use for validating the response (default: None)
"""


T_CreateParamSpec = ParamSpec("T_CreateParamSpec")
T_CreateModel = TypeVar("T_CreateModel", bound=BaseModel)

def wrap_sync_create(
    create: Callable[T_CreateParamSpec, ChatCompletion],
    mode: Mode,
) -> Callable[T_CreateParamSpec, T_CreateModel]:
    def create_sync(
        response_model: Type[T_Model] = None,
        validation_context: dict = None,
        max_retries: int = 1,
        *args: T_CreateParamSpec.args,
        **kwargs: T_CreateParamSpec.kwargs,
    ) -> T_CreateModel:
        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )

        response = retry_sync(
            func=create,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=mode,
        )

        return response

    return create_sync


T_CreateParamSpec = ParamSpec("T_CreateParamSpec")

class InstructorOpenAICompletions(resources.chat.Completions, Generic[T_CreateParamSpec]):
    T_ResponseModel = TypeVar("T_ResponseModel", bound=BaseModel)
    _create: Callable[T_CreateParamSpec, ChatCompletion]

    def __init__(self, openai_completions: resources.chat.Completions, create: Callable[T_CreateParamSpec, ChatCompletion], mode=Mode.FUNCTIONS) -> None:
        self.__dict__.update(openai_completions.__dict__)
        self._openai_completions: resources.chat.Completions = openai_completions
        self._create = create
        self._mode = mode

    # @overload
    # def create(
    #     self,
    #     response_model: Type[T_ResponseModel],
    #     validation_context: dict | None = None,
    #     max_retries: int = 1,
    #     *args: T_CreateParamSpec.args,
    #     **kwargs: T_CreateParamSpec.kwargs,
    # ) -> T_ResponseModel:
    #     ...

    # @overload
    # def create(
    #     self,
    #     response_model: None,
    #     validation_context: dict | None = None,
    #     max_retries: int = 1,
    #     *args: T_CreateParamSpec.args,
    #     **kwargs: T_CreateParamSpec.kwargs,
    # ) -> ChatCompletion:
    #     ...

    def create(
        self,
        response_model: Type[T_ResponseModel] | None = None,
        validation_context: dict | None = None,
        max_retries: int = 1,
        *args: T_CreateParamSpec.args,
        **kwargs: T_CreateParamSpec.kwargs,
    ) -> T_ResponseModel | ChatCompletion:
        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=self._mode, **kwargs
        )

        response = retry_sync(
            func=self._create,
            response_model=response_model,
            # validation_context=validation_context,
            # max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=self._mode,
        )

        return response


T_ChatCreateParams = ParamSpec("T_ChatCreateParams")

class InstructorOpenAIChat(resources.Chat, Generic[T_ChatCreateParams]):
    def __init__(self, openai_chat: resources.Chat, mode: Mode):
        self.__dict__.update(openai_chat.__dict__)
        self._openai_chat: resources.Chat = openai_chat
        self._mode = mode

    @property
    def completions(self):
        return InstructorOpenAICompletions[T_ChatCreateParams](self._openai_chat.completions, self._openai_chat.create, self._mode)


T_OpenAICreateParams = ParamSpec("T_OpenAICreateParams") 

class InstructorOpenAI(OpenAI, Generic[T_OpenAICreateParams]):
    def __init__(self, openai_client: OpenAI, mode=Mode.FUNCTIONS) -> None:
        self.__dict__.update(openai_client.__dict__)
        self._openai_client: OpenAI = openai_client
        self._mode = mode

    @property
    def chat(self):
        return InstructorOpenAIChat[T_OpenAICreateParams](self._openai_client, self._mode)


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
        return InstructorOpenAI(client, create=client.completions.create, mode=mode)


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
