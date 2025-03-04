import { LanguageModelV1Prompt } from "@ai-sdk/provider";
import {
  JsonTestServer,
  StreamingTestServer,
  convertReadableStreamToArray,
} from "@ai-sdk/provider-utils/test";
import { createZhipu } from "./zhipu-provider";

const TEST_PROMPT: LanguageModelV1Prompt = [
  { role: "user", content: [{ type: "text", text: "Hello" }] },
];

const TEST_API_KEY = "70cf6b9542564a69a791632927b63d4e";
const provider = createZhipu({
  apiKey: TEST_API_KEY,
});

const model = provider.chat("glm-4-flash");

describe("doGenerate", () => {
  const server = new JsonTestServer(
    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
  );

  server.setupTestEnvironment();

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
    model = "glm-4-flash-0125",
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
  } = {}) {
    server.responseBodyJson = {
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
    };
  }

  it("should extract text response", async () => {
    prepareJsonResponse({ content: "Hello, World!" });

    const { text } = await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    expect(text).toStrictEqual("Hello, World!");
  });

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
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    expect(usage).toStrictEqual({
      promptTokens: 20,
      completionTokens: 5,
    });
  });

  it("should send request body", async () => {
    prepareJsonResponse({});

    const { request } = await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    expect(request).toStrictEqual({
      body: '{"model":"glm-4-flash","messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}]}',
    });
  });

  it("should send additional response information", async () => {
    prepareJsonResponse({
      id: "test-id",
      created: 123,
      model: "test-model",
    });

    const { response } = await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    expect(response).toStrictEqual({
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
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    expect(usage).toStrictEqual({
      promptTokens: 20,
      completionTokens: undefined,
    });
  });

  it("should extract finish reason", async () => {
    prepareJsonResponse({
      content: "",
      finish_reason: "stop",
    });

    const response = await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
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
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    expect(response.finishReason).toStrictEqual("unknown");
  });

  it("should expose the raw response headers", async () => {
    prepareJsonResponse({ content: "" });

    server.responseHeaders = {
      "test-header": "test-value",
    };

    const { rawResponse } = await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    expect(rawResponse?.headers).toStrictEqual({
      // default headers:
      "content-length": "319",
      "content-type": "application/json",

      // custom header
      "test-header": "test-value",
    });
  });

  it("should pass the model and the messages", async () => {
    prepareJsonResponse({ content: "" });

    await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    expect(await server.getRequestBodyJson()).toStrictEqual({
      model: "glm-4-flash",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    });
  });

  it("should pass settings", async () => {
    prepareJsonResponse();

    await provider
      .chat("glm-4-flash", {
        userId: "test-user-id",
      })
      .doGenerate({
        inputFormat: "prompt",
        mode: { type: "regular" },
        prompt: TEST_PROMPT,
      });

    expect(await server.getRequestBodyJson()).toStrictEqual({
      model: "glm-4-flash",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
      user_id: "test-user-id",
    });
  });

  it("should pass tools and toolChoice", async () => {
    prepareJsonResponse({ content: "" });

    await model.doGenerate({
      inputFormat: "prompt",
      mode: {
        type: "regular",
        tools: [
          {
            type: "function",
            name: "test-tool",
            parameters: {
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
      },
      prompt: TEST_PROMPT,
    });

    expect(await server.getRequestBodyJson()).toStrictEqual({
      model: "glm-4-flash",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
      tools: [
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
      ],
      tool_choice: "auto",
    });
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
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    });

    const requestHeaders = await server.getRequestHeaders();

    expect(requestHeaders).toStrictEqual({
      authorization: `Bearer ${TEST_API_KEY}`,
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    });
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
      inputFormat: "prompt",
      mode: {
        type: "regular",
        tools: [
          {
            type: "function",
            name: "test-tool",
            parameters: {
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
      },
      prompt: TEST_PROMPT,
    });

    expect(result.toolCalls).toStrictEqual([
      {
        args: '{"value":"Spark"}',
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
      },
    ]);
  });

  describe("response format", () => {
    it("should not send a response_format when response format is text", async () => {
      prepareJsonResponse({ content: '{"value":"Spark"}' });

      const model = provider.chat("gpt-4o-2024-08-06");

      await model.doGenerate({
        inputFormat: "prompt",
        mode: { type: "regular" },
        prompt: TEST_PROMPT,
        responseFormat: { type: "text" },
      });

      expect(await server.getRequestBodyJson()).toStrictEqual({
        model: "gpt-4o-2024-08-06",
        messages: [
          { role: "user", content: [{ type: "text", text: "Hello" }] },
        ],
      });
    });

    it('should forward json response format as "json_object" without schema', async () => {
      prepareJsonResponse({ content: '{"value":"Spark"}' });

      const model = provider.chat("gpt-4o-2024-08-06");

      await model.doGenerate({
        inputFormat: "prompt",
        mode: { type: "regular" },
        prompt: TEST_PROMPT,
        responseFormat: { type: "json" },
      });

      expect(await server.getRequestBodyJson()).toStrictEqual({
        model: "gpt-4o-2024-08-06",
        messages: [
          { role: "user", content: [{ type: "text", text: "Hello" }] },
        ],
        response_format: { type: "json_object" },
      });
    });
  });
});

describe("doStream", () => {
  const server = new StreamingTestServer(
    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
  );

  server.setupTestEnvironment();

  function prepareStreamResponse({
    content,
    usage = {
      prompt_tokens: 17,
      total_tokens: 244,
      completion_tokens: 227,
    },
    finish_reason = "stop",
    model = "glm-4-flash-0613",
  }: {
    content: string[];
    usage?: {
      prompt_tokens: number;
      total_tokens: number;
      completion_tokens: number;
    };
    finish_reason?: string;
    model?: string;
  }) {
    server.responseChunks = [
      `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1702657020,"model":"${model}",` +
        `"system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n`,
      ...content.map((text) => {
        return (
          `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1702657020,"model":"${model}",` +
          `"system_fingerprint":null,"choices":[{"index":1,"delta":{"content":"${text}"},"finish_reason":null}]}\n\n`
        );
      }),
      `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1702657020,"model":"${model}",` +
        `"system_fingerprint":null,"choices":[{"index":0,"delta":{},"finish_reason":"${finish_reason}"}]}\n\n`,
      `data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1702657020,"model":"${model}",` +
        `"system_fingerprint":"fp_3bc1b5746c","choices":[],"usage":${JSON.stringify(
          usage,
        )}}\n\n`,
      "data: [DONE]\n\n",
    ];
  }

  it("should stream text deltas", async () => {
    prepareStreamResponse({
      content: ["Hello", ", ", "World!"],
      finish_reason: "stop",
      usage: {
        prompt_tokens: 17,
        total_tokens: 244,
        completion_tokens: 227,
      },
    });

    const { stream } = await model.doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    // note: space moved to last chunk bc of trimming
    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        type: "response-metadata",
        id: "chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP",
        modelId: "glm-4-flash-0613",
        timestamp: new Date("2023-12-15T16:17:00.000Z"),
      },
      { type: "text-delta", textDelta: "" },
      { type: "text-delta", textDelta: "Hello" },
      { type: "text-delta", textDelta: ", " },
      { type: "text-delta", textDelta: "World!" },
      {
        type: "finish",
        finishReason: "stop",
        usage: { promptTokens: 17, completionTokens: 227 },
      },
    ]);
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
    server.responseChunks = [`data: {unparsable}\n\n`, "data: [DONE]\n\n"];

    const { stream } = await model.doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    const elements = await convertReadableStreamToArray(stream);

    expect(elements.length).toBe(2);
    expect(elements[0].type).toBe("error");
    expect(elements[1]).toStrictEqual({
      finishReason: "error",
      type: "finish",
      usage: {
        completionTokens: NaN,
        promptTokens: NaN,
      },
    });
  });

  it("should send request body", async () => {
    prepareStreamResponse({ content: [] });

    const { request } = await model.doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    expect(request).toStrictEqual({
      body: '{"model":"glm-4-flash","messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}],"stream":true}',
    });
  });

  it("should expose the raw response headers", async () => {
    prepareStreamResponse({ content: [] });

    server.responseHeaders = {
      "test-header": "test-value",
    };

    const { rawResponse } = await model.doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    expect(rawResponse?.headers).toStrictEqual({
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
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    });

    expect(await server.getRequestBodyJson()).toStrictEqual({
      stream: true,
      model: "glm-4-flash",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    });
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
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    });

    const requestHeaders = await server.getRequestHeaders();

    expect(requestHeaders).toStrictEqual({
      authorization: `Bearer ${TEST_API_KEY}`,
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    });
  });
});
