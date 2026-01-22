import { ImageModelV2, ImageModelV2CallWarning } from "@ai-sdk/provider";
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

export class ZhipuImageModel implements ImageModelV2 {
  readonly specificationVersion = "v2";
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
    providerOptions,
    headers,
    abortSignal,
  }: Parameters<ImageModelV2["doGenerate"]>[0]): Promise<
    Awaited<ReturnType<ImageModelV2["doGenerate"]>>
  > {
    const warnings: Array<ImageModelV2CallWarning> = [];

    const zhipuProviderOptions = providerOptions
      ? (providerOptions.zhipu as ZhipuImageProviderOptions) ?? {}
      : {};

    if (n != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "n",
        details: "This model does not support multiple images per call.",
      });
    }

    if (aspectRatio != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "aspectRatio",
        details:
          "This model does not support aspect ratio. Use `size` instead.",
      });
    }

    if (seed != null) {
      warnings.push({ type: "unsupported-setting", setting: "seed" });
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

     // Fetch binary content from each image URL
    const images = await Promise.all(
      typedResponse.data.map(async (item) => {
        const imageResponse = await fetch(item.url, { signal: abortSignal });
        const arrayBuffer = await imageResponse.arrayBuffer();
        return new Uint8Array(arrayBuffer);
      }),
    );

    return {
      images: images,
      warnings,
      providerMetadata: {
        zhipu: {
          images: typedResponse.data.map((item) => {
            return {
              url: item.url,
            };
          }),
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

// minimal version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const zhipuImageResponseSchema = z.object({
  created: z.number(),
  data: z.array(z.object({ url: z.url() })),
});
