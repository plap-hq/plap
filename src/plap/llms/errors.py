class ChatCompletionError(Exception):
    pass


class ChatCompletionProviderError(ChatCompletionError):
    pass


class ChatCompletionRateLimitError(ChatCompletionProviderError):
    pass


class ChatCompletionAuthenticationError(ChatCompletionProviderError):
    pass


class ChatCompletionInvalidRequestError(ChatCompletionProviderError):
    pass


class ChatCompletionUnsupportedRequestError(ChatCompletionError):
    pass
