import {
  InvalidResponseDataError,
  LanguageModelV2,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
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

type ZhipuChatConfig = {
  provider: string;
  baseURL: string;
  isMultiModel?: boolean;
  isReasoningModel?: boolean;
  headers: () => Record<string, string | undefined>;
  fetch?: FetchFunction;
};

export class ZhipuChatLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v2" as const;
  readonly defaultObjectGenerationMode = "json";
  readonly supportedUrls: Record<string, RegExp[]> = {
    "image/*": [/^data:image\/[a-zA-Z]+;base64,/, /^https?:\/\/.+$/i],
    "video/*": [/^https?:\/\/.+\.(mp4|webm|ogg)$/i],
  };

  readonly modelId: ZhipuChatModelId;
  readonly settings: ZhipuChatSettings;

  private readonly config: ZhipuChatConfig;

  /**
   * Constructs a new ZhipuChatLanguageModel.
   * @param modelId - The model identifier.
   * @param settings - Settings for the chat.
   * @param config - Model configuration.
   */
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
      this.modelId.includes("z") || this.modelId.includes("thinking");
  }

  /**
   * Getter for the provider name.
   */
  get provider(): string {
    return this.config.provider;
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
  }: Parameters<LanguageModelV2["doGenerate"]>[0]) {
    // const type = mode.type;

    const warnings: LanguageModelV2CallWarning[] = [];

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

    if (tools && tools.length > 0 && this.config.isMultiModel) {
      warnings.push({
        type: "unsupported-setting",
        setting: "tools",
        details: "Tools are not supported with vision models.",
      });
    }

    if (
      tools &&
      tools.length > 0 &&
      tools.some((tool) => tool.type !== "function")
    ) {
      warnings.push({
        type: "unsupported-setting",
        setting: "tools",
        details: "Provider-defined tools are not implemented",
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
      max_tokens: maxOutputTokens,
      temperature: temperature,
      top_p: topP,

      // response format:
      response_format:
        responseFormat?.type === "json" ? { type: "json_object" } : undefined,

      // messages:
      messages: convertToZhipuChatMessages(prompt),

      // tools:
      tool_choice: toolChoice ?? "auto",
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

      // TODO: add provider-specific tool (web_search|retrieval)
    };

    return {
      args: baseArgs,
      warnings,
    };
  }

  async doGenerate(
    options: Parameters<LanguageModelV2["doGenerate"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doGenerate"]>>> {
    const { args, warnings } = this.getArgs(options);

    const {
      value: response,
      rawValue: rawResponse,
      responseHeaders,
    } = await postJsonToApi({
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

    const responseData = response as z.infer<typeof zhipuChatResponseSchema>;
    const choice = responseData.choices[0];

    const content: LanguageModelV2Content[] = [];

    // Extract text content
    const responseText = responseData.choices[0].message.content;
    const responseReasoningText =
      responseData.choices[0].message.reasoning_content;
    if (responseText) {
      if (this.config.isReasoningModel && responseText.includes("<think>")) {
        content.push(
          {
            type: "reasoning",
            text: responseText.split("<think>")[1].split("</think>")[0],
          },
          {
            type: "text",
            text: responseText.split("</think>")[1],
          },
        );
      } else if (this.config.isReasoningModel && responseReasoningText) {
        content.push({
          type: "reasoning",
          text: responseReasoningText,
        });
        content.push({
          type: "text",
          text: responseText,
        });
      } else {
        content.push({
          type: "text",
          text: responseText,
        });
      }
    }

    // Extract tool calls
    if (responseData.choices[0].message.tool_calls) {
      for (const toolCall of responseData.choices[0].message.tool_calls) {
        content.push({
          type: "tool-call",
          toolCallId: toolCall.id,
          toolName: toolCall.function.name,
          input: toolCall.function.arguments,
          providerExecuted: toolCall.type === "function" ? false : true,
        });
      }
    }

    return {
      content,
      finishReason: mapZhipuFinishReason(choice.finish_reason),
      usage: {
        totalTokens: responseData.usage?.total_tokens ?? NaN,
        inputTokens: responseData.usage.prompt_tokens,
        outputTokens: responseData.usage.completion_tokens ?? NaN,
      },
      request: { body: args },
      response: {
        ...getResponseMetadata(responseData),
        headers: responseHeaders,
        body: rawResponse,
      },
      warnings,
    };
  }

  async doStream(
    options: Parameters<LanguageModelV2["doStream"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doStream"]>>> {
    const { args } = this.getArgs(options);

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

    // const { messages: rawPrompt } = args;

    const toolCalls: Array<{
      id: string;
      type: "function";
      function: {
        name: string;
        arguments: string;
      };
      hasFinished: boolean;
    }> = [];

    let finishReason: LanguageModelV2FinishReason = "unknown";
    const usage: LanguageModelV2Usage = {
      inputTokens: undefined,
      outputTokens: undefined,
      totalTokens: undefined,
    };
    let isFirstChunk = true;
    let isActiveReasoning = false;
    let isActiveText = false;

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof zhipuChatChunkSchema>>,
          LanguageModelV2StreamPart
        >({
          transform(chunk, controller) {
            // Emit raw chunk if requested (before anything else)
            if (options.includeRawChunks) {
              controller.enqueue({ type: "raw", rawValue: chunk.rawValue });
            }

            // handle failed chunk parsing / validation:
            if (chunk.success == false) {
              finishReason = "error";
              controller.enqueue({ type: "error", error: chunk.error });
              return;
            }

            const value = chunk.value;

            // handle error chunks:
            if ("error" in value) {
              finishReason = "error";
              controller.enqueue({ type: "error", error: value.error });
              return;
            }

            if (isFirstChunk) {
              isFirstChunk = false;

              controller.enqueue({
                type: "response-metadata",
                ...getResponseMetadata(value),
              });

              controller.enqueue({
                type: "text-start",
                id: "0",
              });
            }

            if (value.usage != null) {
              usage.inputTokens = value.usage.prompt_tokens ?? undefined;
              usage.outputTokens = value.usage.completion_tokens ?? undefined;
              usage.totalTokens = value.usage.total_tokens ?? undefined;
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

                  controller.enqueue({
                    type: "tool-input-start",
                    id: toolCallDelta.id,
                    toolName: toolCallDelta.function.name,
                  });

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
                        type: "tool-input-start",
                        id: toolCall.id,
                        toolName: toolCall.function.name,
                      });
                    }

                    // check if tool call is complete
                    // (some providers send the full tool call in one chunk):
                    if (isParsableJson(toolCall.function.arguments)) {
                      controller.enqueue({
                        type: "tool-input-end",
                        id: toolCall.id,
                      });

                      controller.enqueue({
                        type: "tool-call",
                        toolCallId: toolCall.id ?? generateId(),
                        toolName: toolCall.function.name,
                        input: toolCall.function.arguments,
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
                  type: "tool-input-delta",
                  id: toolCall.id,
                  delta: toolCallDelta.function.arguments ?? "",
                });

                // check if tool call is complete
                if (
                  toolCall.function?.name != null &&
                  toolCall.function?.arguments != null &&
                  isParsableJson(toolCall.function.arguments)
                ) {
                  controller.enqueue({
                    type: "tool-input-end",
                    id: toolCall.id,
                  });

                  controller.enqueue({
                    type: "tool-call",
                    toolCallId: toolCall.id ?? generateId(),
                    toolName: toolCall.function.name,
                    input: toolCall.function.arguments,
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
        reasoning_content: z.string().nullish(),
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
