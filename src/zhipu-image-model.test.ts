import { describe, expect, it, vi } from "vitest";
import { FetchFunction } from "@ai-sdk/provider-utils";
import { ZhipuImageModel } from "./zhipu-image-model";

describe("ZhipuImageModel", () => {
  it("should implement V4 calls and use the configured fetch for image downloads", async () => {
    const fetch: FetchFunction = vi.fn(async (input: RequestInfo | URL) => {
      if (input.toString().endsWith("/images/generations")) {
        return Response.json({
          created: 1,
          data: [{ url: "https://images.example.com/generated.png" }],
        });
      }

      return new Response(new Uint8Array([1, 2, 3]));
    });
    const model = new ZhipuImageModel("glm-image", {
      provider: "zhipu.image",
      url: ({ path }) => `https://api.example.com${path}`,
      headers: () => ({ Authorization: "Bearer test-api-key" }),
      fetch,
      _internal: { currentDate: () => new Date(1) },
    });

    const result = await model.doGenerate({
      prompt: "A yellow bird",
      n: 1,
      size: "1024x1024",
      aspectRatio: undefined,
      seed: undefined,
      files: [
        {
          type: "file",
          mediaType: "image/png",
          data: new Uint8Array([0]),
        },
      ],
      mask: {
        type: "file",
        mediaType: "image/png",
        data: new Uint8Array([0]),
      },
      providerOptions: {},
    });

    expect(model.specificationVersion).toBe("v4");
    expect(result.images).toStrictEqual([new Uint8Array([1, 2, 3])]);
    expect(result.warnings).toEqual(
      expect.arrayContaining([
        { type: "unsupported", feature: "files" },
        { type: "unsupported", feature: "mask" },
      ]),
    );
    expect(fetch).toHaveBeenCalledWith(
      "https://images.example.com/generated.png",
      expect.anything(),
    );
  });
});
