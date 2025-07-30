import {
  EmbeddingModelV2,
  TooManyEmbeddingValuesForCallError,
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createJsonResponseHandler,
  FetchFunction,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { z } from "zod";
import {
  ZhipuEmbeddingModelId,
  ZhipuEmbeddingSettings,
} from "./zhipu-embedding-settings";
import { zhipuFailedResponseHandler } from "./zhipu-error";

type ZhipuEmbeddingConfig = {
  /**
  Override the maximum number of embeddings per call.
   */
  maxEmbeddingsPerCall?: number;

  /**
  Override the parallelism of embedding calls.
   */
  supportsParallelCalls?: boolean;

  provider: string;
  baseURL: string;
  headers: () => Record<string, string | undefined>;
  fetch?: FetchFunction;
};

export class ZhipuEmbeddingModel implements EmbeddingModelV2<string> {
  readonly specificationVersion = "v2" as const;
  readonly modelId: ZhipuEmbeddingModelId;

  private readonly config: ZhipuEmbeddingConfig;
  private readonly settings: ZhipuEmbeddingSettings;

  get provider(): string {
    return this.config.provider;
  }

  get maxEmbeddingsPerCall(): number {
    return this.config.maxEmbeddingsPerCall ?? 64;
  }

  get supportsParallelCalls(): boolean {
    return this.config.supportsParallelCalls ?? true;
  }

  constructor(
    modelId: ZhipuEmbeddingModelId,
    settings: ZhipuEmbeddingSettings,
    config: ZhipuEmbeddingConfig,
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  async doEmbed({
    values,
    abortSignal,
    headers,
  }: Parameters<EmbeddingModelV2<string>["doEmbed"]>[0]): Promise<
    Awaited<ReturnType<EmbeddingModelV2<string>["doEmbed"]>>
  > {
    if (values.length > this.maxEmbeddingsPerCall) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall: this.maxEmbeddingsPerCall,
        values,
      });
    }

    const { responseHeaders, value: response } = await postJsonToApi({
      url: `${this.config.baseURL}/embeddings`,
      headers: combineHeaders(this.config.headers(), headers),
      body: {
        model: this.modelId,
        input: values,
        dimension: this.settings.dimensions,
      },
      failedResponseHandler: zhipuFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        ZhipuTextEmbeddingResponseSchema,
      ),
      abortSignal,
      fetch: this.config.fetch,
    });

    const typedResponse = response as z.infer<
      typeof ZhipuTextEmbeddingResponseSchema
    >;

    return {
      embeddings: typedResponse.data.map((item) => item.embedding),
      usage: typedResponse.usage
        ? { tokens: typedResponse.usage.prompt_tokens }
        : undefined,
      response: { headers: responseHeaders },
    };
  }
}

// minimal version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const ZhipuTextEmbeddingResponseSchema = z.object({
  data: z.array(
    z.object({
      embedding: z.array(z.number()),
      index: z.number().nullish(),
      object: z.string().nullish(),
    }),
  ),
  usage: z
    .object({
      prompt_tokens: z.number(),
      total_tokens: z.number().nullish(),
    })
    .nullish(),
});
