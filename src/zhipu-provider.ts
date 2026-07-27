import {
  EmbeddingModelV4,
  ImageModelV4,
  LanguageModelV4,
  ProviderV4,
} from "@ai-sdk/provider";
import {
  FetchFunction,
  loadApiKey,
  withoutTrailingSlash,
} from "@ai-sdk/provider-utils";
import { ZhipuChatLanguageModel } from "./zhipu-chat-language-model";
import { ZhipuChatModelId, ZhipuChatSettings } from "./zhipu-chat-settings";
import { ZhipuEmbeddingModel } from "./zhipu-embedding-model";
import {
  ZhipuEmbeddingModelId,
  ZhipuEmbeddingSettings,
} from "./zhipu-embedding-settings";
import { ZhipuImageModel } from "./zhipu-image-model";
import { ZhipuImageModelId } from "./zhipu-image-options";

export interface ZhipuProvider extends ProviderV4 {
  (modelId: ZhipuChatModelId, settings?: ZhipuChatSettings): LanguageModelV4;

  languageModel(
    modelId: ZhipuChatModelId,
    settings?: ZhipuChatSettings,
  ): LanguageModelV4;

  chat(
    modelId: ZhipuChatModelId,
    settings?: ZhipuChatSettings,
  ): LanguageModelV4;

  /** @deprecated Use embedding or embeddingModel instead */
  textEmbeddingModel(
    modelId: ZhipuEmbeddingModelId,
    settings?: ZhipuEmbeddingSettings,
  ): EmbeddingModelV4;

  embedding(
    modelId: ZhipuEmbeddingModelId,
    settings?: ZhipuEmbeddingSettings,
  ): EmbeddingModelV4;

  embeddingModel(
    modelId: ZhipuEmbeddingModelId,
    settings?: ZhipuEmbeddingSettings,
  ): EmbeddingModelV4;

  image(modelId: ZhipuImageModelId): ImageModelV4;

  imageModel(modelId: ZhipuImageModelId): ImageModelV4;
}

export interface ZhipuProviderSettings {
  baseURL?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  fetch?: FetchFunction;
}

export function createZhipu(
  options: ZhipuProviderSettings = {},
): ZhipuProvider {
  const baseURL =
    withoutTrailingSlash(options.baseURL) ??
    "https://open.bigmodel.cn/api/paas/v4";

  const getHeaders = () => ({
    Authorization: `Bearer ${loadApiKey({
      apiKey: options.apiKey,
      environmentVariableName: "ZHIPU_API_KEY",
      description: "ZHIPU API key",
    })}`,
    ...options.headers,
  });

  const createChatModel = (
    modelId: ZhipuChatModelId,
    settings: ZhipuChatSettings = {},
  ) =>
    new ZhipuChatLanguageModel(modelId, settings, {
      provider: "zhipu.chat",
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const createEmbeddingModel = (
    modelId: ZhipuEmbeddingModelId,
    settings: ZhipuEmbeddingSettings = {},
  ) =>
    new ZhipuEmbeddingModel(modelId, settings, {
      provider: "zhipu.embedding",
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const createImageModel = (modelId: ZhipuImageModelId) =>
    new ZhipuImageModel(modelId, {
      provider: "zhipu.image",
      url: ({ path }) => `${baseURL}${path}`,
      headers: getHeaders,
      fetch: options.fetch,
      _internal: {
        currentDate: () => new Date(),
      },
    });

  const provider = Object.assign(
    function (modelId: ZhipuChatModelId, settings?: ZhipuChatSettings) {
      if (new.target) {
        throw new Error(
          "The Zhipu model function cannot be called with the new keyword.",
        );
      }

      return createChatModel(modelId, settings);
    } as unknown as ZhipuProvider,
    {
      specificationVersion: "v4" as const,
      languageModel: createChatModel,
      chat: createChatModel,
      /** @deprecated Use embedding or embeddingModel instead */
      textEmbeddingModel: createEmbeddingModel,
      embedding: createEmbeddingModel,
      embeddingModel: createEmbeddingModel,
      image: createImageModel,
      imageModel: createImageModel,
    },
  );

  return provider;
}

export const zhipu = createZhipu();
export const zai = createZhipu({ baseURL: "https://api.z.ai/api/paas/v4" });
