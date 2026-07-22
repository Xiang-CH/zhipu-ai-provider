import { describe, expect, it } from "vitest";
import { convertReadableStreamToArray } from "@ai-sdk/provider-utils/test";
import { createTestServer } from "./test-server";
import { createZhipu } from "./zhipu-provider";

type ZhipuRequestBody = {
  model: string;
  messages: unknown[];
  user_id?: string;
  temperature?: number;
  tools?: unknown[];
  tool_choice?: string;
  stream?: boolean;
  response_format?: { type: string };
  reasoning_effort?: string;
};

const TEST_PROMPT = [
  {
    role: "user" as const,
    content: [{ type: "text" as const, text: "Hello" }],
  },
];

const server = createTestServer({
  "https://open.bigmodel.cn/api/paas/v4/chat/completions": {
    response: { type: "json-value", body: {} },
  },
});

const provider = createZhipu({ apiKey: "test-api-key" });
const model = provider.chat("glm-4-flash");

describe("doGenerate", () => {
  function prepareJsonResponse({
    content = "",
    reasoning_content,
    tool_calls,
    usage = {
      prompt_tokens: 4,
      total_tokens: 34,
      completion_tokens: 30,
    },
    finish_reason = "stop",
    web_search,
    id = "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
    created = 1711115037,
    model = "glm-4-flash",
    headers,
  }: {
    content?: string;
    reasoning_content?: string;
    tool_calls?: Array<{
      id: string;
      type: "function";
      function: {
        name: string;
        arguments: string;
      };
    }>;
    usage?: {
      prompt_tokens: number;
      total_tokens?: number;
      completion_tokens?: number;
    };
    finish_reason?: string;
    web_search?: unknown;
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
        id,
        created,
        model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content,
              reasoning_content,
              tool_calls,
            },
            finish_reason,
          },
        ],
        usage,
        web_search,
      },
    };
  }

  it("should expose text content and V3 finish/usage shapes", async () => {
    prepareJsonResponse({ content: "Hello, World!" });

    const result = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(result.content).toStrictEqual([
      {
        type: "text",
        text: "Hello, World!",
      },
    ]);
    expect(result.finishReason).toStrictEqual({
      unified: "stop",
      raw: "stop",
    });
    expect(result.usage).toStrictEqual({
      inputTokens: {
        total: 4,
        noCache: undefined,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: {
        total: 30,
        text: 30,
        reasoning: undefined,
      },
      raw: {
        prompt_tokens: 4,
        completion_tokens: 30,
        total_tokens: 34,
      },
    });
  });

  it("should expose reasoning, tool calls, and sources", async () => {
    prepareJsonResponse({
      content: "Answer",
      reasoning_content: "Think",
      tool_calls: [
        {
          id: "tool-1",
          type: "function",
          function: {
            name: "lookup",
            arguments: '{"q":"weather"}',
          },
        },
      ],
      web_search: {
        icon: "icon",
        title: "Source title",
        link: "https://example.com",
        media: "web",
        content: "body",
      },
    });

    const result = await model.doGenerate({ prompt: TEST_PROMPT });

    expect(result.content).toStrictEqual([
      { type: "reasoning", text: "Think" },
      { type: "text", text: "Answer" },
      {
        type: "tool-call",
        toolCallId: "tool-1",
        toolName: "lookup",
        input: '{"q":"weather"}',
        providerExecuted: false,
      },
      {
        type: "source",
        sourceType: "url",
        id: "https://example.com",
        url: "https://example.com",
        title: "Source title",
        providerMetadata: {
          zhipu: {
            icon: "icon",
            media: "web",
            content: "body",
          },
        },
      },
    ]);
  });

  it("should send request body and response metadata", async () => {
    prepareJsonResponse({
      id: "test-id",
      created: 123,
      model: "test-model",
    });

    const result = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(result.request?.body).toMatchObject({
      model: "glm-4-flash",
      messages: [{ role: "user", content: "Hello" }],
      tool_choice: "auto",
    });
    expect(result.response).toMatchObject({
      id: "test-id",
      timestamp: new Date(123 * 1000),
      modelId: "test-model",
    });
  });

  it("should map unsupported settings to V3 warnings", async () => {
    prepareJsonResponse({ content: "" });

    const result = await model.doGenerate({
      prompt: TEST_PROMPT,
      topK: 3,
      seed: 7,
      toolChoice: { type: "required" },
      responseFormat: {
        type: "json",
        schema: { type: "object" },
      },
    });

    expect(result.warnings).toEqual(
      expect.arrayContaining([
        { type: "unsupported", feature: "topK", details: undefined },
        { type: "unsupported", feature: "seed", details: undefined },
        {
          type: "unsupported",
          feature: "toolChoice",
          details: "Only 'auto' tool choice is supported",
        },
        {
          type: "unsupported",
          feature: "responseFormat",
          details:
            "Structured output with schema is not supported, use json response format instead.",
        },
      ]),
    );
  });

  it("should warn when non-vision models receive non-text user parts", async () => {
    prepareJsonResponse({ content: "" });

    const result = await model.doGenerate({
      prompt: [
        { role: "system", content: "You are concise." },
        {
          role: "user",
          content: [
            { type: "text", text: "Describe this image" },
            {
              type: "file",
              data: new URL("https://example.com/image.png"),
              mediaType: "image/png",
            },
          ],
        },
      ],
    });

    expect(result.warnings).toContainEqual({
      type: "other",
      message: "Non-vision models does not support message parts",
    });
  });

  it("should pass providerOptions.zhipu and override base args", async () => {
    prepareJsonResponse({ content: "" });

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

    const calls =
      server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"]
        .calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
    expect(body.user_id).toBe("override-user-id");
    expect(body.temperature).toBe(0.8);
  });

  it.each([
    "max",
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
    "none",
  ] as const)("should map %s reasoning effort to the API request", async (reasoningEffort) => {
    prepareJsonResponse({ content: "" });

    await provider
      .chat("glm-5.2", { reasoningEffort })
      .doGenerate({ prompt: TEST_PROMPT });

    const calls =
      server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"]
        .calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
    expect(body.reasoning_effort).toBe(reasoningEffort);
  });

  it("should prefer request-level reasoningEffort and omit its camel-case form", async () => {
    prepareJsonResponse({ content: "" });

    await provider
      .chat("glm-5.2", { reasoningEffort: "low" })
      .doGenerate({
        prompt: TEST_PROMPT,
        providerOptions: {
          zhipu: { reasoningEffort: "high" },
        },
      });

    const calls =
      server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"]
        .calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
    expect(body.reasoning_effort).toBe("high");
    expect(body).not.toHaveProperty("reasoningEffort");
  });

  it("should preserve raw reasoning_effort provider-option passthrough", async () => {
    prepareJsonResponse({ content: "" });

    await provider
      .chat("glm-5.2", { reasoningEffort: "low" })
      .doGenerate({
        prompt: TEST_PROMPT,
        providerOptions: {
          zhipu: { reasoning_effort: "minimal" },
        },
      });

    const calls =
      server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"]
        .calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
    expect(body.reasoning_effort).toBe("minimal");
  });

  it.each(["glm-5.1", "glm-5", "glm-4.7-flash"])(
    "should warn when reasoning effort is used with %s",
    async (modelId) => {
      prepareJsonResponse({ content: "" });

      const result = await provider
        .chat(modelId, { reasoningEffort: "high" })
        .doGenerate({ prompt: TEST_PROMPT });

      expect(result.warnings).toContainEqual({
        type: "other",
        message: `reasoning_effort is not supported by model "${modelId}" and will likely be ignored by the upstream API.`,
      });
    },
  );

  it("should warn when providerOptions set reasoningEffort for an unsupported model", async () => {
    prepareJsonResponse({ content: "" });

    const result = await provider.chat("glm-5.1").doGenerate({
      prompt: TEST_PROMPT,
      providerOptions: {
        zhipu: { reasoningEffort: "high" },
      },
    });

    expect(result.warnings).toContainEqual({
      type: "other",
      message:
        'reasoning_effort is not supported by model "glm-5.1" and will likely be ignored by the upstream API.',
    });
  });

  it.each(["glm-5.2", "glm-5.3", "glm-next"])(
    "should not warn for supported or unrecognized future model %s",
    async (modelId) => {
      prepareJsonResponse({ content: "" });

      const result = await provider
        .chat(modelId, { reasoningEffort: "high" })
        .doGenerate({ prompt: TEST_PROMPT });

      expect(result.warnings).not.toContainEqual(
        expect.objectContaining({
          message: expect.stringContaining("reasoning_effort is not supported"),
        }),
      );
    },
  );

  it("should not warn when reasoning effort is omitted", async () => {
    prepareJsonResponse({ content: "" });

    const result = await provider.chat("glm-5.1").doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(result.warnings).not.toContainEqual(
      expect.objectContaining({
        message: expect.stringContaining("reasoning_effort is not supported"),
      }),
    );
  });
});

describe("doStream", () => {
  it("should stream V3 events", async () => {
    server.urls[
      "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    ].response = {
      type: "stream-chunks",
      chunks: [
        'data: {"id":"1","created":1,"model":"glm-4-flash","choices":[{"index":0,"delta":{"content":"Hel","reasoning_content":"Think ","tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\\"q\\":\\"" }}]},"finish_reason":null}],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}\n\n',
        'data: {"id":"1","created":1,"model":"glm-4-flash","choices":[{"index":0,"delta":{"content":"lo","tool_calls":[{"index":0,"function":{"arguments":"weather\\"}"}}]},"finish_reason":"tool_calls"}],"web_search":{"icon":"icon","title":"Source title","link":"https://example.com","media":"web","content":"body"}}\n\n',
        "data: [DONE]\n\n",
      ],
    };

    const result = await model.doStream({
      prompt: TEST_PROMPT,
    });

    const events = await convertReadableStreamToArray(result.stream);

    expect(events).toEqual(
      expect.arrayContaining([
        { type: "stream-start", warnings: [] },
        {
          type: "response-metadata",
          id: "1",
          modelId: "glm-4-flash",
          timestamp: new Date(1000),
        },
        { type: "reasoning-start", id: "reasoning-0" },
        { type: "reasoning-delta", id: "reasoning-0", delta: "Think " },
        { type: "text-start", id: "txt-0" },
        { type: "text-delta", id: "txt-0", delta: "Hel" },
        { type: "text-delta", id: "txt-0", delta: "lo" },
        {
          type: "tool-input-start",
          id: "call-1",
          toolName: "lookup",
          providerExecuted: false,
        },
        { type: "tool-input-delta", id: "call-1", delta: '{"q":"' },
        { type: "tool-input-delta", id: "call-1", delta: 'weather"}' },
        { type: "tool-input-end", id: "call-1" },
        {
          type: "tool-call",
          toolCallId: "call-1",
          toolName: "lookup",
          input: '{"q":"weather"}',
        },
        {
          type: "source",
          sourceType: "url",
          id: "https://example.com",
          url: "https://example.com",
          title: "Source title",
          providerMetadata: {
            zhipu: {
              icon: "icon",
              media: "web",
              content: "body",
            },
          },
        },
        { type: "reasoning-end", id: "reasoning-0" },
        { type: "text-end", id: "txt-0" },
        {
          type: "finish",
          finishReason: {
            unified: "tool-calls",
            raw: "tool_calls",
          },
          usage: {
            inputTokens: {
              total: 4,
              noCache: undefined,
              cacheRead: undefined,
              cacheWrite: undefined,
            },
            outputTokens: {
              total: 3,
              text: 3,
              reasoning: undefined,
            },
            raw: {
              prompt_tokens: 4,
              completion_tokens: 3,
              total_tokens: 7,
            },
          },
        },
      ]),
    );
  });

  it("should map reasoningEffort in streaming requests", async () => {
    server.urls[
      "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    ].response = {
      type: "stream-chunks",
      chunks: ["data: [DONE]\\n\\n"],
    };

    await provider.chat("glm-5.2").doStream({
      prompt: TEST_PROMPT,
      providerOptions: {
        zhipu: { reasoningEffort: "high" },
      },
    });

    const calls =
      server.urls["https://open.bigmodel.cn/api/paas/v4/chat/completions"]
        .calls;
    const body = calls[calls.length - 1].requestBodyJson as ZhipuRequestBody;
    expect(body.reasoning_effort).toBe("high");
    expect(body.stream).toBe(true);
    expect(body).not.toHaveProperty("reasoningEffort");
  });
});
