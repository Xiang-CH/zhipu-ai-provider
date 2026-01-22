import {
  EmbeddingModelV2,
  LanguageModelV2,
  ProviderV2,
} from "@ai-sdk/provider";
import {
  FetchFunction,
  loadApiKey,
  withoutTrailingSlash,
} from "@ai-sdk/provider-utils";
import { ZhipuChatLanguageModel } from "./zhipu-chat-language-model";
import { ZhipuChatModelId, ZhipuChatSettings } from "./zhipu-chat-settings";
import { ZhipuEmbeddingModel } from "./zhipu-embedding-model";
import { ZhipuImageModelId } from "./zhipu-image-options";
import { ZhipuImageModel } from "./zhipu-image-model";
import {
  ZhipuEmbeddingModelId,
  ZhipuEmbeddingSettings,
} from "./zhipu-embedding-settings";

export interface ZhipuProvider extends ProviderV2 {
  (modelId: ZhipuChatModelId, settings?: ZhipuChatSettings): LanguageModelV2;

  /**
Creates a model for text generation.
*/
  languageModel(
    modelId: ZhipuChatModelId,
    settings?: ZhipuChatSettings,
  ): LanguageModelV2;

  /**
Creates a model for text generation.
*/
  chat(
    modelId: ZhipuChatModelId,
    settings?: ZhipuChatSettings,
  ): LanguageModelV2;

  /**
Creates a model for text embedding.
*/
  textEmbeddingModel: (
    modelId: ZhipuEmbeddingModelId,
    settings?: ZhipuEmbeddingSettings,
  ) => EmbeddingModelV2<string>;
}

export interface ZhipuProviderSettings {
  /**
Use a different URL prefix for API calls, e.g. to use proxy servers.
The default prefix is `https://open.bigmodel.cn/api/paas/v4`.
   */
  baseURL?: string;

  /**
API key that is being send using the `Authorization` header.
It defaults to the `ZHIPU_API_KEY` environment variable.
   */
  apiKey?: string;

  /**
Custom headers to include in the requests.
     */
  headers?: Record<string, string>;

  /**
Custom fetch implementation. You can use it as a middleware to intercept requests,
or to provide a custom fetch implementation for e.g. testing.
    */
  fetch?: FetchFunction;
}

/**
Create a Zhipu AI provider instance.
 */
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

  const provider = function (
    modelId: ZhipuChatModelId,
    settings?: ZhipuChatSettings,
  ) {
    if (new.target) {
      throw new Error(
        "The Zhipu model function cannot be called with the new keyword.",
      );
    }

    return createChatModel(modelId, settings);
  };

  provider.languageModel = createChatModel;
  provider.chat = createChatModel;

  provider.textEmbeddingModel = createEmbeddingModel;

  provider.image = createImageModel;
  provider.imageModel = createImageModel;

  return provider;
}

/**
Default Zhipu provider instance for bigmodel.cn API.
 */
export const zhipu = createZhipu();

/**
Default Z.ai provider instance for z.ai API.
 */
export const zai = createZhipu({ baseURL: "https://api.z.ai/api/paas/v4" });
