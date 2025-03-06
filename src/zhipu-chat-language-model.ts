import {
  InvalidResponseDataError,
  LanguageModelV1,
  LanguageModelV1CallWarning,
  LanguageModelV1FinishReason,
  LanguageModelV1StreamPart,
} from "@ai-sdk/provider";
import {
  isParsableJson,
  generateId,
  FetchFunction,
  ParseResult,
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { z } from "zod";
import { convertToZhipuChatMessages } from "./convert-to-zhipu-chat-messages";
import { mapZhipuFinishReason } from "./map-zhipu-finish-reason";
import { ZhipuChatModelId, ZhipuChatSettings } from "./zhipu-chat-settings";
import { zhipuFailedResponseHandler } from "./zhipu-error";
import { getResponseMetadata } from "./get-response-metadata";
import { prepareTools } from "./zhipu-prepare-tools";

type ZhipuChatConfig = {
  provider: string;
  baseURL: string;
  isMultiModel: boolean;
  isReasoningModel: boolean;
  headers: () => Record<string, string | undefined>;
  fetch?: FetchFunction;
  
};

export class ZhipuChatLanguageModel implements LanguageModelV1 {
  readonly specificationVersion = "v1";
  readonly defaultObjectGenerationMode = "json";
  readonly supportsImageUrls = false;

  readonly modelId: ZhipuChatModelId;
  readonly settings: ZhipuChatSettings;

  private readonly config: ZhipuChatConfig;

  /**
   * Constructs a new QwenChatLanguageModel.
   * @param modelId - The model identifier.
   * @param settings - Settings for the chat.
   * @param config - Model configuration.
   */
  constructor(
    modelId: ZhipuChatModelId,
    settings: ZhipuChatSettings,
    config: ZhipuChatConfig,
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
    this.config.isMultiModel = this.modelId.includes("4v");
    this.config.isReasoningModel = this.modelId.includes("zero");
  }

  /**
   * Getter for the provider name.
   */
  get provider(): string {
    return this.config.provider;
  }

  private getArgs({
    mode,
    prompt,
    maxTokens,
    temperature,
    topP,
    topK,
    frequencyPenalty,
    presencePenalty,
    stopSequences,
    responseFormat,
    seed,
  }: Parameters<LanguageModelV1["doGenerate"]>[0]) {
    const type = mode.type;

    const warnings: LanguageModelV1CallWarning[] = [];

    if (!this.config.isMultiModel && prompt.every(msg => msg.role === "user" && !msg.content.every(part => part.type === 'text'))) {
      warnings.push({
        type: "other",
        message: "Non-vision models does not support message parts",
      });
    }

    if (topK != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "topK",
      });
    }

    if (frequencyPenalty != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "frequencyPenalty",
      });
    }

    if (presencePenalty != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "presencePenalty",
      });
    }

    if (stopSequences != null && this.config.isMultiModel) {
      warnings.push({
        type: "unsupported-setting",
        setting: "stopSequences",
        details: "Stop sequences are not supported for vision model",
      });
    }

    if (stopSequences != null && stopSequences.length > 1) {
      warnings.push({
        type: "unsupported-setting",
        setting: "stopSequences",
        details: "Only supports one stop sequence",
      });
    }

    if (seed != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "seed",
      });
    }

    if (
      responseFormat != null &&
      responseFormat.type === "json" &&
      (this.config.isMultiModel || this.config.isReasoningModel)
    ) {
      warnings.push({
        type: "unsupported-setting",
        setting: "responseFormat",
        details:
          "JSON response format is not supported with vision and reasoning models.",
      });
    }

    if (
      responseFormat != null &&
      responseFormat.type === "json" &&
      responseFormat.schema != null
    ) {
      warnings.push({
        type: "unsupported-setting",
        setting: "responseFormat",
        details:
          "JSON response format schema is only supported with structuredOutputs, provide the schema.",
      });
    }

    const baseArgs = {
      // model id:
      model: this.modelId,

      // model specific settings:
      user_id: this.settings.userId,
      do_sample: this.settings.doSample,
      request_id: this.settings.requestId,

      // standardized settings:
      max_tokens: maxTokens,
      temperature: temperature,
      top_p: topP,

      // response format:
      response_format:
        responseFormat?.type === "json" ? { type: "json_object" } : undefined,

      // messages:
      messages: convertToZhipuChatMessages(prompt),
    };

    switch (type) {
      case "regular": {
        const { tools, tool_choice, toolWarnings } = prepareTools(mode);

        return {
          args: { ...baseArgs, tools, tool_choice },
          warnings: [...warnings, ...toolWarnings],
        };
      }

      case "object-json": {
        return {
          args: {
            ...baseArgs,
            response_format: { type: "json_object" },
          },
          warnings,
        };
      }

      case "object-tool": {
        return {
          args: {
            ...baseArgs,
            tool_choice: "auto",
            tools: [mode.tool],
          },
          warnings,
        };
      }

      default: {
        const _exhaustiveCheck: never = type;
        throw new Error(`Unsupported type: ${_exhaustiveCheck}`);
      }
    }
  }

  async doGenerate(
    options: Parameters<LanguageModelV1["doGenerate"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV1["doGenerate"]>>> {
    const { args, warnings } = this.getArgs(options);

    const { responseHeaders, value: response } = await postJsonToApi({
      url: `${this.config.baseURL}/chat/completions`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body: args,
      failedResponseHandler: zhipuFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        zhipuChatResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const { messages: rawPrompt, ...rawSettings } = args;
    const choice = response.choices[0];

    return {
      text: choice.message.content ?? undefined,
      toolCalls: choice.message.tool_calls?.map((toolCall) => ({
        toolCallType: "function",
        toolCallId: toolCall.id,
        toolName: toolCall.function.name,
        args: toolCall.function.arguments!,
      })),
      finishReason: mapZhipuFinishReason(choice.finish_reason),
      usage: {
        promptTokens: response.usage.prompt_tokens,
        completionTokens: response.usage.completion_tokens ?? NaN,
      },
      rawCall: { rawPrompt, rawSettings },
      rawResponse: { headers: responseHeaders },
      request: { body: JSON.stringify(args) },
      response: getResponseMetadata(response),
      warnings,
    };
  }

  async doStream(
    options: Parameters<LanguageModelV1["doStream"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV1["doStream"]>>> {
    const { args, warnings } = this.getArgs(options);

    const body = { ...args, stream: true };
    // const metadataExtractor = this.config.metadataExtractor?.createStreamExtractor();

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

    const { messages: rawPrompt, ...rawSettings } = args;

    const toolCalls: Array<{
      id: string;
      type: "function";
      function: {
        name: string;
        arguments: string;
      };
      hasFinished: boolean;
    }> = [];

    let finishReason: LanguageModelV1FinishReason = "unknown";
    let usage: {
      promptTokens: number | undefined;
      completionTokens: number | undefined;
    } = {
      promptTokens: undefined,
      completionTokens: undefined,
    };
    let isFirstChunk = true;

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof zhipuChatChunkSchema>>,
          LanguageModelV1StreamPart
        >({
          transform(chunk, controller) {
            // handle failed chunk parsing / validation:
            if (chunk.success == false) {
              finishReason = "error";
              controller.enqueue({ type: "error", error: chunk.error });
              return;
            }

            const value = chunk.value;

            // handle error chunks:
            // if ('error' in value) {
            //   finishReason = 'error';
            //   controller.enqueue({ type: 'error', error: value.error });
            //   return;
            // }

            if (isFirstChunk) {
              isFirstChunk = false;

              controller.enqueue({
                type: "response-metadata",
                ...getResponseMetadata(value),
              });
            }

            if (value.usage != null) {
              const { prompt_tokens, completion_tokens } = value.usage;

              usage = {
                promptTokens: prompt_tokens ?? undefined,
                completionTokens: completion_tokens ?? undefined,
              };
            }

            const choice = value.choices[0];

            if (choice?.finish_reason != null) {
              if (choice.finish_reason === "network_error") {
                controller.enqueue({
                  type: "error",
                  error: new Error(`Error: Network Error`),
                });
                return;
              }

              finishReason = mapZhipuFinishReason(choice.finish_reason);
            }

            if (choice?.delta == null) {
              return;
            }

            const delta = choice.delta;

            if (delta.content != null) {
              controller.enqueue({
                type: "text-delta",
                textDelta: delta.content,
              });
            }

            const mappedToolCalls = delta.tool_calls;

            if (mappedToolCalls != null) {
              for (const toolCallDelta of mappedToolCalls) {
                const index = toolCallDelta.index;

                // Tool call start. OpenAI returns all information except the arguments in the first chunk.
                if (toolCalls[index] == null) {
                  if (toolCallDelta.type !== "function") {
                    throw new InvalidResponseDataError({
                      data: toolCallDelta,
                      message: `Expected 'function' type.`,
                    });
                  }

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
                    type: "function",
                    function: {
                      name: toolCallDelta.function.name,
                      arguments: toolCallDelta.function.arguments ?? "",
                    },
                    hasFinished: false,
                  };

                  const toolCall = toolCalls[index];

                  if (
                    toolCall.function?.name != null &&
                    toolCall.function?.arguments != null
                  ) {
                    // send delta if the argument text has already started:
                    if (toolCall.function.arguments.length > 0) {
                      controller.enqueue({
                        type: "tool-call-delta",
                        toolCallType: "function",
                        toolCallId: toolCall.id,
                        toolName: toolCall.function.name,
                        argsTextDelta: toolCall.function.arguments,
                      });
                    }

                    // check if tool call is complete
                    // (some providers send the full tool call in one chunk):
                    if (isParsableJson(toolCall.function.arguments)) {
                      controller.enqueue({
                        type: "tool-call",
                        toolCallType: "function",
                        toolCallId: toolCall.id ?? generateId(),
                        toolName: toolCall.function.name,
                        args: toolCall.function.arguments,
                      });
                      toolCall.hasFinished = true;
                    }
                  }

                  continue;
                }

                // existing tool call, merge if not finished
                const toolCall = toolCalls[index];

                if (toolCall.hasFinished) {
                  continue;
                }

                if (toolCallDelta.function?.arguments != null) {
                  toolCall.function!.arguments +=
                    toolCallDelta.function?.arguments ?? "";
                }

                // send delta
                controller.enqueue({
                  type: "tool-call-delta",
                  toolCallType: "function",
                  toolCallId: toolCall.id,
                  toolName: toolCall.function.name,
                  argsTextDelta: toolCallDelta.function.arguments ?? "",
                });

                // check if tool call is complete
                if (
                  toolCall.function?.name != null &&
                  toolCall.function?.arguments != null &&
                  isParsableJson(toolCall.function.arguments)
                ) {
                  controller.enqueue({
                    type: "tool-call",
                    toolCallType: "function",
                    toolCallId: toolCall.id ?? generateId(),
                    toolName: toolCall.function.name,
                    args: toolCall.function.arguments,
                  });
                  toolCall.hasFinished = true;
                }
              }
            }
          },

          flush(controller) {
            controller.enqueue({
              type: "finish",
              finishReason,
              usage: {
                promptTokens: usage.promptTokens ?? NaN,
                completionTokens: usage.completionTokens ?? NaN,
              },
            });
          },
        }),
      ),
      rawCall: { rawPrompt, rawSettings },
      rawResponse: { headers: responseHeaders },
      request: { body: JSON.stringify(body) },
      warnings,
    };
  }
}

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const zhipuChatResponseSchema = z.object({
  id: z.string().nullish(),
  created: z.number().nullish(),
  model: z.string().nullish(),
  choices: z.array(
    z.object({
      message: z.object({
        role: z.literal("assistant"),
        content: z.string().nullish(),
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
    .object({
      icon: z.string(),
      title: z.string(),
      link: z.string(),
      media: z.string(),
      content: z.string(),
    })
    .nullish(),
});

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const zhipuChatChunkSchema = z.object({
  id: z.string().nullish(),
  created: z.number().nullish(),
  model: z.string().nullish(),
  choices: z.array(
    z.object({
      delta: z.object({
        role: z.enum(["assistant"]).optional(),
        content: z.string().nullish(),
        tool_calls: z
          .array(
            z.object({
              id: z.string(),
              index: z.number(),
              type: z.literal("function"),
              function: z.object({ name: z.string(), arguments: z.string() }),
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
    .object({
      icon: z.string(),
      title: z.string(),
      link: z.string(),
      media: z.string(),
      content: z.string(),
    })
    .nullish(),
});
// zhipuErrorDataSchema
