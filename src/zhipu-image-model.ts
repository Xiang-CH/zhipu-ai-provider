import { ImageModelV4 } from "@ai-sdk/provider";
import {
  combineHeaders,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  FetchFunction,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { z } from "zod";
import { defaultZhipuErrorStructure } from "./zhipu-error";
import {
  ZhipuImageModelId,
  ZhipuImageProviderOptions,
  sizeSchema,
} from "./zhipu-image-options";

export type ZhipuImageModelConfig = {
  provider: string;
  headers: () => Record<string, string | undefined>;
  url: (options: { modelId: string; path: string }) => string;
  fetch?: FetchFunction;
  _internal?: {
    currentDate?: () => Date;
  };
};

export class ZhipuImageModel implements ImageModelV4 {
  readonly specificationVersion = "v4" as const;
  readonly maxImagesPerCall = 10;

  get provider(): string {
    return this.config.provider;
  }

  constructor(
    readonly modelId: ZhipuImageModelId,
    private readonly config: ZhipuImageModelConfig,
  ) {}

  async doGenerate({
    prompt,
    n,
    size,
    aspectRatio,
    seed,
    files,
    mask,
    providerOptions,
    headers,
    abortSignal,
  }: Parameters<ImageModelV4["doGenerate"]>[0]): Promise<
    Awaited<ReturnType<ImageModelV4["doGenerate"]>>
  > {
    const warnings: Array<{
      type: "unsupported";
      feature: string;
      details?: string;
    }> = [];

    const zhipuProviderOptions = providerOptions
      ? (providerOptions.zhipu as ZhipuImageProviderOptions) ?? {}
      : {};

    if (n !== 1) {
      warnings.push({
        type: "unsupported",
        feature: "n",
        details: "This model does not support multiple images per call.",
      });
    }

    if (aspectRatio != null) {
      warnings.push({
        type: "unsupported",
        feature: "aspectRatio",
        details:
          "This model does not support aspect ratio. Use `size` instead.",
      });
    }

    if (seed != null) {
      warnings.push({ type: "unsupported", feature: "seed" });
    }

    if (files != null && files.length > 0) {
      warnings.push({ type: "unsupported", feature: "files" });
    }

    if (mask != null) {
      warnings.push({ type: "unsupported", feature: "mask" });
    }

    if (
      size != null &&
      !sizeSchema.safeParse({
        width: parseInt(size.split("x")[0]),
        height: parseInt(size.split("x")[1]),
      }).success
    ) {
      throw new Error(
        "Invalid size. Size must be an object with width and height, both divisible by 16, and within the range of 512 to 2048 pixels.",
      );
    }

    const currentDate = this.config._internal?.currentDate?.() ?? new Date();
    const { value: response, responseHeaders } = await postJsonToApi({
      url: this.config.url({
        path: "/images/generations",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), headers),
      body: {
        model: this.modelId,
        prompt,
        size,
        ...(zhipuProviderOptions ?? {}),
      },
      failedResponseHandler: createJsonErrorResponseHandler(
        defaultZhipuErrorStructure,
      ),
      successfulResponseHandler: createJsonResponseHandler(
        zhipuImageResponseSchema,
      ),
      abortSignal,
      fetch: this.config.fetch,
    });

    const typedResponse = response as z.infer<typeof zhipuImageResponseSchema>;

    const images = await Promise.all(
      typedResponse.data.map(async (item) => {
        const imageResponse = await (this.config.fetch ?? fetch)(item.url, {
          signal: abortSignal,
        });
        const arrayBuffer = await imageResponse.arrayBuffer();
        return new Uint8Array(arrayBuffer);
      }),
    );

    return {
      images,
      warnings,
      providerMetadata: {
        zhipu: {
          images: typedResponse.data.map((item) => ({ url: item.url })),
        },
      },
      response: {
        timestamp: currentDate,
        modelId: this.modelId,
        headers: responseHeaders,
      },
    };
  }
}

const zhipuImageResponseSchema = z.object({
  created: z.number(),
  data: z.array(z.object({ url: z.url() })),
});
