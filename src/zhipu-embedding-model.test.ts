import { describe, expect, it } from "vitest";
import { EmbeddingModelV2Embedding } from "@ai-sdk/provider";
import { createZhipu } from "./zhipu-provider";
import { createTestServer } from "./test-server";

const dummyEmbeddings = [
  [0.1, 0.2, 0.3, 0.4, 0.5],
  [0.6, 0.7, 0.8, 0.9, 1.0],
];
const testValues = ["sunny day at the beach", "rainy day in the city"];

const provider = createZhipu({ apiKey: "test-api-key" });
const model = provider.embeddingModel("embedding-3");

describe("doEmbed", () => {
  const server = createTestServer({
    "https://open.bigmodel.cn/api/paas/v4/embeddings": {},
  });

  function prepareJsonResponse({
    embeddings = dummyEmbeddings,
    usage = { prompt_tokens: 8, total_tokens: 8 },
    headers,
  }: {
    embeddings?: EmbeddingModelV2Embedding[];
    usage?: { prompt_tokens: number; total_tokens: number };
    headers?: Record<string, string>;
  } = {}) {
    server.urls["https://open.bigmodel.cn/api/paas/v4/embeddings"].response = {
      type: "json-value",
      headers,
      body: {
        id: "b322cfc2b9d34e2f8e14fc99874faee5",
        object: "list",
        data: embeddings.map((embedding, i) => ({
          object: "embedding",
          embedding,
          index: i,
        })),
        model: "embedding-3",
        usage,
      },
    };
  }

  it("should extract embedding", async () => {
    prepareJsonResponse();

    const { embeddings } = await model.doEmbed({ values: testValues });

    expect(embeddings).toStrictEqual(dummyEmbeddings);
  });

  it("should extract usage", async () => {
    prepareJsonResponse({
      usage: { prompt_tokens: 20, total_tokens: 20 },
    });

    const { usage } = await model.doEmbed({ values: testValues });

    expect(usage).toStrictEqual({ tokens: 20 });
  });

  it("should expose the raw response headers", async () => {
    prepareJsonResponse({
      headers: {
        "test-header": "test-value",
      },
    });

    const { response } = await model.doEmbed({ values: testValues });

    expect(response?.headers).toStrictEqual({
      // default headers:
      "content-length": "265",
      "content-type": "application/json",

      // custom header
      "test-header": "test-value",
    });
  });

  it("should pass the model and the values", async () => {
    prepareJsonResponse();

    await model.doEmbed({ values: testValues });

    const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/embeddings"].calls;
    expect(await calls[calls.length - 1].requestBodyJson).toStrictEqual({
      model: "embedding-3",
      input: testValues,
    });
  });

  it("should pass headers", async () => {
    prepareJsonResponse();

    const provider = createZhipu({
      apiKey: "test-api-key",
      headers: {
        "Custom-Provider-Header": "provider-header-value",
      },
    });

    await provider.embeddingModel("embedding-3").doEmbed({
      values: testValues,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    });

    const calls = server.urls["https://open.bigmodel.cn/api/paas/v4/embeddings"].calls;
    const headers = calls[calls.length - 1].requestHeaders as any;
    expect(headers.authorization).toBe("Bearer test-api-key");
    expect(headers["content-type"]).toBe("application/json");
    expect(headers["custom-provider-header"]).toBe("provider-header-value");
    expect(headers["custom-request-header"]).toBe("request-header-value");
  });
});
