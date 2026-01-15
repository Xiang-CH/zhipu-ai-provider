import { describe, it, expect } from "vitest";
import { LanguageModelV2Prompt } from "@ai-sdk/provider";
import { convertReadableStreamToArray } from "@ai-sdk/provider-utils/test";
import { createZhipu } from "./zhipu-provider";
import { createTestServer } from "./test-server";

// Zhipu API request body structure (subset of fields used in tests)
type ZhipuRequestBody = {
  model: string;
  messages: unknown[];
  user_id?: string;
  temperature?: number;
  top_p?: number;
  tools?: unknown[];
  tool_choice?: string;
  stream?: boolean;
  response_format?: { type: string };
};

const TEST_PROMPT: LanguageModelV2Prompt = [
  { role: "user", content: [{ type: "text", text: "Hello" }] },
];

const TEST_API_KEY = "test-api-key";
const provider = createZhipu({
  apiKey: TEST_API_KEY,
});

const model = provider.chat("glm-4-flash");

const server = createTestServer({
  "https://open.bigmodel.cn/api/paas/v4/chat/completions": {},
});

describe("doGenerate", () => {
  function prepareJsonResponse({
    content = "",
    tool_calls,
    function_call,
    usage = {
      prompt_tokens: 4,
      total_tokens: 34,
      completion_tokens: 30,
    },
    finish_reason = "stop",
    id = "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
    created = 1711115037,
    model = "glm-4-flash",
    headers,
  }: {
    content?: string;
    tool_calls?: Array<{
      id: string;
      type: "function";
      function: {
        name: string;
        arguments: string;
      };
    }>;
    function_call?: {
      name: string;
      arguments: string;
    };
    usage?: {
      prompt_tokens?: number;
      total_tokens?: number;
      completion_tokens?: number;
    };
    finish_reason?: string;
    created?: number;
    id?: string;
    model?: string;
    headers?: Record<string, string>;
  } = {}) {
    server.urls[
      "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    ].response = {
      type: "json-value",
      headers,
      body: {
        headers,
        id,
        object: "chat.completion",
        created,
        model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content,
              tool_calls,
              function_call,
            },
            finish_reason,
          },
        ],
        usage,
        system_fingerprint: "fp_3bc1b5746c",
      },
    };
  }

  it("should extract text response", async () => {
    prepareJsonResponse({ content: "Hello, World!" });

    const { content } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(content).toMatchInlineSnapshot(`
      [
        {
          "text": "Hello, World!",
          "type": "text",
        },
      ]
    `);
  }); // Closing brace for the 'it' block on line 90

  it("should extract usage", async () => {
    prepareJsonResponse({
      content: "",
      usage: {
        prompt_tokens: 20,
        total_tokens: 25,
        completion_tokens: 5,
      },
    });

    const { usage } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(usage).toStrictEqual({
      inputTokens: 20,
      outputTokens: 5,
      totalTokens: 25,
    });
  });

  it("should send request body", async () => {
    prepareJsonResponse({});

    const { request } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(request.body).toMatchObject({
      model: "glm-4-flash",
      messages: [{ role: "user", content: "Hello" }],
    });
  });

  it("should send additional response information", async () => {
    prepareJsonResponse({
      id: "test-id",
      created: 123,
      model: "test-model",
    });

    const { response } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(response).toMatchObject({
      id: "test-id",
      timestamp: new Date(123 * 1000),
      modelId: "test-model",
    });
  });

  it("should support partial usage", async () => {
    prepareJsonResponse({
      content: "",
      usage: { prompt_tokens: 20, total_tokens: 20 },
    });

    const { usage } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(usage).toStrictEqual({
      inputTokens: 20,
      outputTokens: NaN,
      totalTokens: 20,
    });
  });

  it("should extract finish reason", async () => {
    prepareJsonResponse({
      content: "",
      finish_reason: "stop",
    });

    const response = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(response.finishReason).toStrictEqual("stop");
  });

  it("should support unknown finish reason", async () => {
    prepareJsonResponse({
      content: "",
      finish_reason: "eos",
    });

    const response = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(response.finishReason).toStrictEqual("unknown");
  });

  it("should expose the raw response headers", async () => {
    prepareJsonResponse({
      headers: { "test-header": "test-value" },
    });

    const { response } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(response?.headers).toMatchObject({
      "content-type": "application/json",
      "test-header": "test-value",
    });
  });

  it("should pass the model and the messages", async () => {
    prepareJsonResponse({ content: "" });

    await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"].calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
    expect(body.model).toBe("glm-4-flash");
    expect(body.messages).toEqual([{ role: "user", content: "Hello" }]);
  });

  it("should pass settings", async () => {
    prepareJsonResponse();

    await provider
      .chat("glm-4-flash", {
        userId: "test-user-id",
      })
      .doGenerate({
        prompt: TEST_PROMPT,
      });

    const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"].calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
    expect(body.model).toBe("glm-4-flash");
    expect(body.messages).toEqual([{ role: "user", content: "Hello" }]);
    expect(body.user_id).toBe("test-user-id");
  });

  it("should pass providerOptions.zhipu and override base args", async () => {
    prepareJsonResponse();

    await provider
      .chat("glm-4-flash", {
        userId: "test-user-id",
      })
      .doGenerate({
        prompt: TEST_PROMPT,
        providerOptions: {
          zhipu: {
            user_id: "override-user-id",
            temperature: 0.8,
          },
        },
      });

    const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"].calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
    expect(body.model).toBe("glm-4-flash");
    expect(body.messages).toEqual([{ role: "user", content: "Hello" }]);
    expect(body.user_id).toBe("override-user-id");
    expect(body.temperature).toBe(0.8);
    expect(body.tool_choice).toBe("auto");
  });

  it("should pass tools and toolChoice", async () => {
    prepareJsonResponse({ content: "" });

    await model.doGenerate({
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
      toolChoice: {
        type: "tool",
        toolName: "test-tool",
      },
      prompt: TEST_PROMPT,
    });

    const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"].calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
    expect(body.model).toBe("glm-4-flash");
    expect(body.messages).toEqual([{ role: "user", content: "Hello" }]);
    expect(body.tools).toEqual([
      {
        type: "function",
        function: {
          name: "test-tool",
          parameters: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      },
    ]);
    expect(body.tool_choice).toBe("auto");
  });

  it("should pass headers", async () => {
    prepareJsonResponse({ content: "" });

    const provider = createZhipu({
      apiKey: TEST_API_KEY,
      headers: {
        "Custom-Provider-Header": "provider-header-value",
      },
    });

    await provider.chat("glm-4-flash").doGenerate({
      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    });

    const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"].calls;
    const headers = calls[calls.length - 1].requestHeaders as Record<string, string>;
    expect(headers.authorization).toBe(`Bearer ${TEST_API_KEY}`);
    expect(headers["content-type"]).toBe("application/json");
    expect(headers["custom-provider-header"]).toBe("provider-header-value");
    expect(headers["custom-request-header"]).toBe("request-header-value");
  });

  it("should parse tool results", async () => {
    prepareJsonResponse({
      tool_calls: [
        {
          id: "call_O17Uplv4lJvD6DVdIvFFeRMw",
          type: "function",
          function: {
            name: "test-tool",
            arguments: '{"value":"Spark"}',
          },
        },
      ],
    });

    const result = await model.doGenerate({
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
      toolChoice: {
        type: "tool",
        toolName: "test-tool",
      },
      prompt: TEST_PROMPT,
    });

    expect(result.content).toEqual([
      {
        type: "tool-call",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolName: "test-tool",
        input: '{"value":"Spark"}',
        // v6 field: false indicates the provider did not execute the tool; execution is delegated to the client.
        providerExecuted: false,
      },
    ]);
  });

  describe("response format", () => {
    it("should not send a response_format when response format is text", async () => {
      prepareJsonResponse({ content: '{"value":"Spark"}' });

      const model = provider.chat("glm-4-flash");

      await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: { type: "text" },
      });

      const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"].calls;
      expect(calls[calls.length - 1].requestBodyJson).toMatchObject({
        model: "glm-4-flash",
        messages: [{ role: "user", content: "Hello" }],
      });
    });

    it('should forward json response format as "json_object" without schema', async () => {
      prepareJsonResponse({ content: '{"value":"Spark"}' });

      const model = provider.chat("glm-4-flash");

      await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: { type: "json" },
      });

      const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"].calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
      expect(body.model).toBe("glm-4-flash");
      expect(body.messages).toEqual([{ role: "user", content: "Hello" }]);
      expect(body.response_format).toEqual({ type: "json_object" });
    });
  });
});

describe("doStream", () => {
  function prepareStreamResponse({
    content,
    finish_reason = "stop",
    headers,
  }: {
    content: string[];
    finish_reason?: string;
    headers?: Record<string, string>;
  }) {
    server.urls[
      "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    ].response = {
      type: "stream-chunks",
      headers,
      chunks: [
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"grok-beta",` +
          `"system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n`,
        ...content.map((text) => {
          return (
            `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"grok-beta",` +
            `"system_fingerprint":null,"choices":[{"index":1,"delta":{"content":"${text}"},"finish_reason":null}]}\n\n`
          );
        }),
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"grok-beta",` +
          `"system_fingerprint":null,"choices":[{"index":0,"delta":{},"finish_reason":"${finish_reason}"}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"grok-beta",` +
          `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"${finish_reason}"}],` +
          `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,` +
          `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
        "data: [DONE]\n\n",
      ],
    };
  }

  it("should stream text deltas", async () => {
    prepareStreamResponse({
      content: ["Hello", ", ", "World!"],
      finish_reason: "stop",
    });

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    });

    // note: space moved to last chunk bc of trimming
    const result = await convertReadableStreamToArray(stream);

    // Check key elements exist
    expect(result).toContainEqual(
      expect.objectContaining({
        type: "response-metadata",
      }),
    );

    expect(result).toContainEqual(
      expect.objectContaining({
        type: "finish",
        finishReason: "stop",
        usage: {
          inputTokens: 18,
          outputTokens: 439,
          totalTokens: 457,
        },
      }),
    );
  });

  // it("should stream tool deltas", async () => {
  //     server.responseChunks = [
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,` +
  //             `"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments":""}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\""}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"value"}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\":\\""}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Spark"}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"le"}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" Day"}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"}"}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[],"usage":{"prompt_tokens":53,"completion_tokens":17,"total_tokens":70}}\n\n`,
  //         "data: [DONE]\n\n",
  //     ];

  //     const { stream } = await model.doStream({
  //         inputFormat: "prompt",
  //         mode: {
  //             type: "regular",
  //             tools: [
  //                 {
  //                     type: "function",
  //                     name: "test-tool",
  //                     parameters: {
  //                         type: "object",
  //                         properties: { value: { type: "string" } },
  //                         required: ["value"],
  //                         additionalProperties: false,
  //                         $schema: "http://json-schema.org/draft-07/schema#",
  //                     },
  //                 },
  //             ],
  //         },
  //         prompt: TEST_PROMPT,
  //     });

  //     expect(await convertReadableStreamToArray(stream)).toStrictEqual([
  //         {
  //             type: "response-metadata",
  //             id: "chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP",
  //             modelId: "glm-4-flash-0125",
  //             timestamp: new Date("2024-03-25T09:06:38.000Z"),
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: '{"',
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: "value",
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: '":"',
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: "Spark",
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: "le",
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: " Day",
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: '"}',
  //         },
  //         {
  //             type: "tool-call",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             args: '{"value":"Sparkle Day"}',
  //         },
  //         {
  //             type: "finish",
  //             finishReason: "tool-calls",
  //             usage: { promptTokens: 53, completionTokens: 17 },
  //         },
  //     ]);
  // });

  // it("should stream tool call deltas when tool call arguments are passed in the first chunk", async () => {
  //     server.responseChunks = [
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,` +
  //             `"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments":"{\\""}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"va"}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"lue"}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\":\\""}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Spark"}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"le"}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" Day"}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"}"}}]},` +
  //             `"finish_reason":null}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n`,
  //         `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"glm-4-flash-0125",` +
  //             `"system_fingerprint":"fp_3bc1b5746c","choices":[],"usage":{"prompt_tokens":53,"completion_tokens":17,"total_tokens":70}}\n\n`,
  //         "data: [DONE]\n\n",
  //     ];

  //     const { stream } = await model.doStream({
  //         inputFormat: "prompt",
  //         mode: {
  //             type: "regular",
  //             tools: [
  //                 {
  //                     type: "function",
  //                     name: "test-tool",
  //                     parameters: {
  //                         type: "object",
  //                         properties: { value: { type: "string" } },
  //                         required: ["value"],
  //                         additionalProperties: false,
  //                         $schema: "http://json-schema.org/draft-07/schema#",
  //                     },
  //                 },
  //             ],
  //         },
  //         prompt: TEST_PROMPT,
  //     });

  //     expect(await convertReadableStreamToArray(stream)).toStrictEqual([
  //         {
  //             type: "response-metadata",
  //             id: "chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP",
  //             modelId: "glm-4-flash-0125",
  //             timestamp: new Date("2024-03-25T09:06:38.000Z"),
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: '{"',
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: "va",
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: "lue",
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: '":"',
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: "Spark",
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: "le",
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: " Day",
  //         },
  //         {
  //             type: "tool-call-delta",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             argsTextDelta: '"}',
  //         },
  //         {
  //             type: "tool-call",
  //             toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
  //             toolCallType: "function",
  //             toolName: "test-tool",
  //             args: '{"value":"Sparkle Day"}',
  //         },
  //         {
  //             type: "finish",
  //             finishReason: "tool-calls",
  //             usage: { promptTokens: 53, completionTokens: 17 },
  //         },
  //     ]);
  // });

  it("should handle unparsable stream parts", async () => {
    server.urls[
      "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    ].response = {
      type: "stream-chunks",
      chunks: [`data: {unparsable}\n\n`, "data: [DONE]\n\n"],
    };

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    });

    const elements = await convertReadableStreamToArray(stream);

    expect(elements.length).toBe(2);
    expect(elements[0].type).toBe("error");
    expect(elements[1]).toMatchObject({
      finishReason: "error",
      type: "finish",
    });
  });

  it("should send request body", async () => {
    prepareStreamResponse({ content: [] });

    const { request } = await model.doStream({
      prompt: TEST_PROMPT,
    });

    expect(request.body).toMatchObject({
      model: "glm-4-flash",
      messages: [{ role: "user", content: "Hello" }],
      stream: true,
    });
  });

  it("should expose the raw response headers", async () => {
    prepareStreamResponse({
      content: [],
      headers: {
        "test-header": "test-value",
      },
    });

    const { response } = await model.doStream({
      prompt: TEST_PROMPT,
    });

    expect(response?.headers).toStrictEqual({
      // default headers:
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",

      // custom header
      "test-header": "test-value",
    });
  });

  it("should pass the messages and the model", async () => {
    prepareStreamResponse({ content: [] });

    await model.doStream({
      prompt: TEST_PROMPT,
    });

    const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"].calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
    expect(body.stream).toBe(true);
    expect(body.model).toBe("glm-4-flash");
    expect(body.messages).toEqual([{ role: "user", content: "Hello" }]);
  });

  it("should pass providerOptions.zhipu in stream mode", async () => {
    prepareStreamResponse({ content: [] });

    await model.doStream({
      prompt: TEST_PROMPT,
      providerOptions: {
        zhipu: {
          temperature: 0.9,
          top_p: 0.95,
        },
      },
    });

    const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"].calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
    expect(body.stream).toBe(true);
    expect(body.model).toBe("glm-4-flash");
    expect(body.messages).toEqual([{ role: "user", content: "Hello" }]);
    expect(body.temperature).toBe(0.9);
    expect(body.top_p).toBe(0.95);
    expect(body.tool_choice).toBe("auto");
  });

  it("should pass headers", async () => {
    prepareStreamResponse({ content: [] });

    const provider = createZhipu({
      apiKey: TEST_API_KEY,
      headers: {
        "Custom-Provider-Header": "provider-header-value",
      },
    });

    await provider.chat("glm-4-flash").doStream({
      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    });

    const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"].calls;
    const headers = calls[calls.length - 1].requestHeaders as Record<string, string>;
    expect(headers.authorization).toBe(`Bearer ${TEST_API_KEY}`);
    expect(headers["content-type"]).toBe("application/json");
    expect(headers["custom-provider-header"]).toBe("provider-header-value");
    expect(headers["custom-request-header"]).toBe("request-header-value");
  });
});
