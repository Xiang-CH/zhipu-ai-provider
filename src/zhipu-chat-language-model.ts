import {
  InvalidResponseDataError,
  LanguageModelV3,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
  SharedV3Warning,
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  FetchFunction,
  generateId,
  isParsableJson,
  ParseResult,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { z } from "zod";
import { convertToZhipuChatMessages } from "./convert-to-zhipu-chat-messages";
import { getResponseMetadata } from "./get-response-metadata";
import { mapZhipuFinishReason } from "./map-zhipu-finish-reason";
import { ZhipuChatModelId, ZhipuChatSettings } from "./zhipu-chat-settings";
import { zhipuFailedResponseHandler } from "./zhipu-error";

type ZhipuChatConfig = {
  provider: string;
  baseURL: string;
  isMultiModel?: boolean;
  isReasoningModel?: boolean;
  headers: () => Record<string, string | undefined>;
  fetch?: FetchFunction;
};

type ZhipuWebSearchResult = z.infer<typeof zhipuWebSearchItemSchema>;

export class ZhipuChatLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3" as const;
  readonly supportedUrls: Record<string, RegExp[]> = {
    "image/*": [/^data:image\/[a-zA-Z]+;base64,/, /^https?:\/\/.+$/i],
    "video/*": [/^https?:\/\/.+\.(mp4|webm|ogg)$/i],
  };

  readonly modelId: ZhipuChatModelId;
  readonly settings: ZhipuChatSettings;

  private readonly config: ZhipuChatConfig;

  constructor(
    modelId: ZhipuChatModelId,
    settings: ZhipuChatSettings,
    config: ZhipuChatConfig,
  ) {
    this.modelId = modelId.toLocaleLowerCase();
    this.settings = settings;
    this.config = config;
    this.config.isMultiModel = this.modelId.includes("v");
    this.config.isReasoningModel =
      this.modelId.includes("z") ||
      this.modelId.includes("thinking") ||
      settings.thinking?.type === "enabled";
  }

  get provider(): string {
    return this.config.provider;
  }

  private unsupported(
    feature: string,
    details?: string,
  ): Extract<SharedV3Warning, { type: "unsupported" }> {
    return { type: "unsupported", feature, details };
  }

  private getArgs({
    prompt,
    maxOutputTokens,
    temperature,
    topP,
    topK,
    frequencyPenalty,
    presencePenalty,
    stopSequences,
    responseFormat,
    seed,
    tools,
    toolChoice,
  }: Parameters<LanguageModelV3["doGenerate"]>[0]) {
    const warnings: SharedV3Warning[] = [];

    if (
      !this.config.isMultiModel &&
      prompt.every(
        (msg) =>
          msg.role === "user" &&
          !msg.content.every((part) => part.type === "text"),
      )
    ) {
      warnings.push({
        type: "other",
        message: "Non-vision models does not support message parts",
      });
    }

    if (topK != null) {
      warnings.push(this.unsupported("topK"));
    }

    if (frequencyPenalty != null) {
      warnings.push(this.unsupported("frequencyPenalty"));
    }

    if (presencePenalty != null) {
      warnings.push(this.unsupported("presencePenalty"));
    }

    if (stopSequences != null && this.config.isMultiModel) {
      warnings.push(
        this.unsupported(
          "stopSequences",
          "Stop sequences are not supported for vision model",
        ),
      );
    }

    if (stopSequences != null && stopSequences.length > 1) {
      warnings.push(
        this.unsupported("stopSequences", "Only supports one stop sequence"),
      );
    }

    if (seed != null) {
      warnings.push(this.unsupported("seed"));
    }

    if (
      responseFormat &&
      responseFormat.type === "json" &&
      (this.config.isMultiModel || this.config.isReasoningModel)
    ) {
      warnings.push(
        this.unsupported(
          "responseFormat",
          "JSON response format is not supported with vision and reasoning models.",
        ),
      );
    }

    if (tools && tools.length > 0 && this.config.isMultiModel) {
      warnings.push(
        this.unsupported(
          "tools",
          "Tools are not supported with vision models.",
        ),
      );
    }

    if (
      tools &&
      tools.length > 0 &&
      tools.some((tool) => tool.type !== "function")
    ) {
      warnings.push(
        this.unsupported("tools", "Provider-defined tools are not implemented"),
      );
    }

    if (
      responseFormat &&
      responseFormat.type === "json" &&
      responseFormat.schema
    ) {
      warnings.push(
        this.unsupported(
          "responseFormat",
          "Structured output with schema is not supported, use json response format instead.",
        ),
      );
    }

    if (toolChoice?.type != null && toolChoice.type !== "auto") {
      warnings.push(
        this.unsupported("toolChoice", "Only 'auto' tool choice is supported"),
      );
    }

    const baseArgs = {
      model: this.modelId,
      user_id: this.settings.userId,
      do_sample: this.settings.doSample,
      request_id: this.settings.requestId,
      thinking: this.settings.thinking
        ? {
            type: this.settings.thinking.type,
            ...(this.settings.thinking.clearThinking !== undefined && {
              clear_thinking: this.settings.thinking.clearThinking,
            }),
          }
        : undefined,
      max_tokens: maxOutputTokens,
      temperature,
      top_p: topP,
      response_format:
        responseFormat?.type === "json" ? { type: "json_object" } : undefined,
      messages: convertToZhipuChatMessages(prompt),
      tool_choice: "auto",
      tools:
        tools
          ?.filter((tool) => tool.type === "function")
          .map((tool) => ({
            type: "function" as const,
            function: {
              name: tool.name,
              description: tool.description ?? undefined,
              parameters: tool.inputSchema,
            },
          })) ?? undefined,
    };

    return { args: baseArgs, warnings };
  }

  private mapUsage(
    usage:
      | {
          prompt_tokens: number;
          completion_tokens?: number | null;
          total_tokens?: number | null;
        }
      | null
      | undefined,
  ): LanguageModelV3Usage {
    return {
      inputTokens: {
        total: usage?.prompt_tokens,
        noCache: undefined,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: {
        total: usage?.completion_tokens ?? undefined,
        text: usage?.completion_tokens ?? undefined,
        reasoning: undefined,
      },
      raw: usage
        ? {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens ?? undefined,
            total_tokens: usage.total_tokens ?? undefined,
          }
        : undefined,
    };
  }

  private buildSourceContent(
    webSearch: ZhipuWebSearchResult[] | ZhipuWebSearchResult | null | undefined,
  ): LanguageModelV3Content[] {
    const items =
      webSearch == null
        ? []
        : Array.isArray(webSearch)
          ? webSearch
          : [webSearch];

    return items.map((item, index) => ({
      type: "source",
      sourceType: "url",
      id: item.link || `source-${index}`,
      url: item.link,
      title: item.title || undefined,
      providerMetadata: {
        zhipu: {
          icon: item.icon,
          media: item.media,
          content: item.content,
        },
      },
    }));
  }

  async doGenerate(
    options: Parameters<LanguageModelV3["doGenerate"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV3["doGenerate"]>>> {
    const { args, warnings } = this.getArgs(options);
    const providerOptions = options.providerOptions || {};
    const zhipuOptions = providerOptions.zhipu || {};
    const fullArgs = { ...args, ...zhipuOptions };

    const {
      value: response,
      rawValue: rawResponse,
      responseHeaders,
    } = await postJsonToApi({
      url: `${this.config.baseURL}/chat/completions`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body: fullArgs,
      failedResponseHandler: zhipuFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        zhipuChatResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const responseData = response as z.infer<typeof zhipuChatResponseSchema>;
    const choice = responseData.choices[0];
    const content: LanguageModelV3Content[] = [];
    const responseText = choice.message.content;
    const responseReasoningText = choice.message.reasoning_content;

    if (responseReasoningText) {
      content.push({
        type: "reasoning",
        text: responseReasoningText,
      });
    }

    if (responseText) {
      if (this.config.isReasoningModel && responseText.includes("<think>")) {
        const reasoningText = responseText
          .split("<think>")[1]
          ?.split("</think>")[0];
        const text = responseText.split("</think>")[1];

        if (reasoningText) {
          content.push({ type: "reasoning", text: reasoningText });
        }

        if (text) {
          content.push({ type: "text", text });
        }
      } else {
        content.push({
          type: "text",
          text: responseText,
        });
      }
    }

    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        content.push({
          type: "tool-call",
          toolCallId: toolCall.id,
          toolName: toolCall.function.name,
          input: toolCall.function.arguments,
          providerExecuted: false,
        });
      }
    }

    content.push(...this.buildSourceContent(responseData.web_search));

    return {
      content,
      finishReason: mapZhipuFinishReason(choice.finish_reason),
      usage: this.mapUsage(responseData.usage),
      request: { body: fullArgs },
      response: {
        ...getResponseMetadata(responseData),
        headers: responseHeaders,
        body: rawResponse,
      },
      warnings,
    };
  }

  async doStream(
    options: Parameters<LanguageModelV3["doStream"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV3["doStream"]>>> {
    const { args, warnings } = this.getArgs(options);
    const providerOptions = options.providerOptions || {};
    const zhipuOptions = providerOptions.zhipu || {};
    const body = { ...args, ...zhipuOptions, stream: true };

    const { responseHeaders, value: response } = await postJsonToApi({
      url: `${this.config.baseURL}/chat/completions`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: zhipuFailedResponseHandler,
      successfulResponseHandler:
        createEventSourceResponseHandler(zhipuChatChunkSchema),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const toolCalls: Array<{
      id: string;
      toolName: string;
      input: string;
      hasFinished: boolean;
    }> = [];

    let finishReason: LanguageModelV3FinishReason = {
      unified: "other",
      raw: undefined,
    };
    let usage: LanguageModelV3Usage = this.mapUsage(undefined);
    let isFirstChunk = true;
    let isActiveReasoning = false;
    let isActiveText = false;

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof zhipuChatChunkSchema>>,
          LanguageModelV3StreamPart
        >({
          transform(chunk, controller) {
            if (options.includeRawChunks) {
              controller.enqueue({ type: "raw", rawValue: chunk.rawValue });
            }

            if (!chunk.success) {
              finishReason = { unified: "error", raw: "parse_error" };
              controller.enqueue({ type: "error", error: chunk.error });
              return;
            }

            const value = chunk.value;

            if (isFirstChunk) {
              isFirstChunk = false;
              controller.enqueue({
                type: "stream-start",
                warnings,
              });
              controller.enqueue({
                type: "response-metadata",
                ...getResponseMetadata(value),
              });
            }

            if ("error" in value) {
              finishReason = { unified: "error", raw: "provider_error" };
              controller.enqueue({ type: "error", error: value.error });
              return;
            }

            if (value.usage != null) {
              usage = {
                inputTokens: {
                  total: value.usage.prompt_tokens ?? undefined,
                  noCache: undefined,
                  cacheRead: undefined,
                  cacheWrite: undefined,
                },
                outputTokens: {
                  total: value.usage.completion_tokens ?? undefined,
                  text: value.usage.completion_tokens ?? undefined,
                  reasoning: undefined,
                },
                raw: {
                  prompt_tokens: value.usage.prompt_tokens,
                  completion_tokens: value.usage.completion_tokens ?? undefined,
                  total_tokens: value.usage.total_tokens ?? undefined,
                },
              };
            }

            const choice = value.choices[0];

            if (choice?.finish_reason != null) {
              if (choice.finish_reason === "network_error") {
                finishReason = { unified: "error", raw: choice.finish_reason };
                controller.enqueue({
                  type: "error",
                  error: new Error("Error: Network Error"),
                });
                return;
              }

              finishReason = mapZhipuFinishReason(choice.finish_reason);
            }

            if (choice?.delta == null) {
              return;
            }

            const delta = choice.delta;

            if (delta.reasoning_content != null) {
              if (!isActiveReasoning) {
                controller.enqueue({
                  type: "reasoning-start",
                  id: "reasoning-0",
                });
                isActiveReasoning = true;
              }

              controller.enqueue({
                id: "reasoning-0",
                type: "reasoning-delta",
                delta: delta.reasoning_content,
              });
            }

            if (delta.content != null) {
              if (!isActiveText) {
                controller.enqueue({ type: "text-start", id: "txt-0" });
                isActiveText = true;
              }

              controller.enqueue({
                id: "txt-0",
                type: "text-delta",
                delta: delta.content,
              });
            }

            if (delta.tool_calls != null) {
              for (const toolCallDelta of delta.tool_calls) {
                const index = toolCallDelta.index;

                if (toolCalls[index] == null) {
                  if (toolCallDelta.id == null) {
                    throw new InvalidResponseDataError({
                      data: toolCallDelta,
                      message: `Expected 'id' to be a string.`,
                    });
                  }

                  if (toolCallDelta.function?.name == null) {
                    throw new InvalidResponseDataError({
                      data: toolCallDelta,
                      message: `Expected 'function.name' to be a string.`,
                    });
                  }

                  toolCalls[index] = {
                    id: toolCallDelta.id,
                    toolName: toolCallDelta.function.name,
                    input: toolCallDelta.function.arguments ?? "",
                    hasFinished: false,
                  };

                  controller.enqueue({
                    type: "tool-input-start",
                    id: toolCallDelta.id,
                    toolName: toolCallDelta.function.name,
                    providerExecuted: false,
                  });

                  if (toolCallDelta.function.arguments) {
                    controller.enqueue({
                      type: "tool-input-delta",
                      id: toolCallDelta.id,
                      delta: toolCallDelta.function.arguments,
                    });
                  }

                  if (isParsableJson(toolCalls[index].input)) {
                    controller.enqueue({
                      type: "tool-input-end",
                      id: toolCalls[index].id,
                    });
                    controller.enqueue({
                      type: "tool-call",
                      toolCallId: toolCalls[index].id ?? generateId(),
                      toolName: toolCalls[index].toolName,
                      input: toolCalls[index].input,
                    });
                    toolCalls[index].hasFinished = true;
                  }

                  continue;
                }

                const toolCall = toolCalls[index];

                if (toolCall.hasFinished) {
                  continue;
                }

                if (toolCallDelta.function?.arguments != null) {
                  toolCall.input += toolCallDelta.function.arguments;
                }

                controller.enqueue({
                  type: "tool-input-delta",
                  id: toolCall.id,
                  delta: toolCallDelta.function?.arguments ?? "",
                });

                if (isParsableJson(toolCall.input)) {
                  controller.enqueue({
                    type: "tool-input-end",
                    id: toolCall.id,
                  });
                  controller.enqueue({
                    type: "tool-call",
                    toolCallId: toolCall.id ?? generateId(),
                    toolName: toolCall.toolName,
                    input: toolCall.input,
                  });
                  toolCall.hasFinished = true;
                }
              }
            }

            const sources = Array.isArray(value.web_search)
              ? value.web_search
              : value.web_search != null
                ? [value.web_search]
                : [];

            for (const [index, source] of sources.entries()) {
              controller.enqueue({
                type: "source",
                sourceType: "url",
                id: source.link || `source-${index}`,
                url: source.link,
                title: source.title || undefined,
                providerMetadata: {
                  zhipu: {
                    icon: source.icon,
                    media: source.media,
                    content: source.content,
                  },
                },
              });
            }
          },
          flush(controller) {
            if (isActiveReasoning) {
              controller.enqueue({
                type: "reasoning-end",
                id: "reasoning-0",
              });
            }

            if (isActiveText) {
              controller.enqueue({
                type: "text-end",
                id: "txt-0",
              });
            }

            controller.enqueue({
              type: "finish",
              finishReason,
              usage,
            });
          },
        }),
      ),
      request: { body },
      response: { headers: responseHeaders },
    };
  }
}

const zhipuWebSearchItemSchema = z.object({
  icon: z.string().catch(""),
  title: z.string().catch(""),
  link: z.string().catch(""),
  media: z.string().catch(""),
  content: z.string().catch(""),
});

const zhipuChatResponseSchema = z.object({
  id: z.string().nullish(),
  created: z.number().nullish(),
  model: z.string().nullish(),
  choices: z.array(
    z.object({
      message: z.object({
        role: z.literal("assistant"),
        content: z.string().nullish(),
        reasoning_content: z.string().nullish(),
        tool_calls: z
          .array(
            z.object({
              id: z.string(),
              index: z.number().nullish(),
              type: z.literal("function"),
              function: z.object({ name: z.string(), arguments: z.string() }),
            }),
          )
          .nullish(),
      }),
      index: z.number(),
      finish_reason: z.string().nullish(),
    }),
  ),
  usage: z.object({
    prompt_tokens: z.number(),
    completion_tokens: z.number().nullish(),
    total_tokens: z.number().nullish(),
  }),
  web_search: z
    .union([zhipuWebSearchItemSchema, z.array(zhipuWebSearchItemSchema)])
    .nullish(),
});

const zhipuChatChunkSchema = z.object({
  id: z.string().nullish(),
  created: z.number().nullish(),
  model: z.string().nullish(),
  choices: z.array(
    z.object({
      delta: z.object({
        role: z.enum(["assistant"]).optional(),
        content: z.string().nullish(),
        reasoning_content: z.string().nullish(),
        tool_calls: z
          .array(
            z.object({
              id: z.string().optional(),
              index: z.number(),
              type: z.literal("function").optional(),
              function: z
                .object({
                  name: z.string().optional(),
                  arguments: z.string().optional(),
                })
                .optional(),
            }),
          )
          .nullish(),
      }),
      finish_reason: z.string().nullish(),
      index: z.number(),
    }),
  ),
  usage: z
    .object({
      prompt_tokens: z.number(),
      completion_tokens: z.number().nullish(),
      total_tokens: z.number().nullish(),
    })
    .nullish(),
  web_search: z
    .union([zhipuWebSearchItemSchema, z.array(zhipuWebSearchItemSchema)])
    .nullish(),
});
