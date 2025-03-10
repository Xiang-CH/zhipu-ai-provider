import { describe, expect, it } from "vitest";
import { EmbeddingModelV1Embedding } from "@ai-sdk/provider";
import { JsonTestServer } from "@ai-sdk/provider-utils/test";
import { createZhipu } from "./zhipu-provider";

const dummyEmbeddings = [
  [0.1, 0.2, 0.3, 0.4, 0.5],
  [0.6, 0.7, 0.8, 0.9, 1.0],
];
const testValues = ["sunny day at the beach", "rainy day in the city"];

const provider = createZhipu({ apiKey: "test-api-key" });
const model = provider.textEmbeddingModel("embedding-3");

describe("doEmbed", () => {
  const server = new JsonTestServer(
    "https://open.bigmodel.cn/api/paas/v4/embeddings",
  );

  server.setupTestEnvironment();

  function prepareJsonResponse({
    embeddings = dummyEmbeddings,
    usage = { prompt_tokens: 8, total_tokens: 8 },
  }: {
    embeddings?: EmbeddingModelV1Embedding[];
    usage?: { prompt_tokens: number; total_tokens: number };
  } = {}) {
    server.responseBodyJson = {
      id: "b322cfc2b9d34e2f8e14fc99874faee5",
      object: "list",
      data: embeddings.map((embedding, i) => ({
        object: "embedding",
        embedding,
        index: i,
      })),
      model: "embedding-3",
      usage,
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
    prepareJsonResponse();

    server.responseHeaders = {
      "test-header": "test-value",
    };

    const { rawResponse } = await model.doEmbed({ values: testValues });

    expect(rawResponse?.headers).toStrictEqual({
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

    expect(await server.getRequestBodyJson()).toStrictEqual({
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

    await provider.embedding("embedding-3").doEmbed({
      values: testValues,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    });

    const requestHeaders = await server.getRequestHeaders();

    expect(requestHeaders).toStrictEqual({
      authorization: "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    });
  });
});
