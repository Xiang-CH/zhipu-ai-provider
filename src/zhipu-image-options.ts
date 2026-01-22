import { z } from "zod";

// https://bigmodel.cn/dev/api/vector/embedding
export type ZhipuImageModelId =
  | "glm-image"
  | "cogview-3-flash"
  | "cogview-4"
  | (string & {});

const sizeSchema = z
  .object({
    width: z
      .number()
      .min(512, "Width must be at least 512px")
      .max(2048, "Width must be at most 2048px")
      .refine((val) => val % 16 === 0, "Width must be divisible by 16"),
    height: z
      .number()
      .min(512, "Height must be at least 512px")
      .max(2048, "Height must be at most 2048px")
      .refine((val) => val % 16 === 0, "Height must be divisible by 16"),
  })
  .refine(
    ({ width, height }) => width * height <= Math.pow(2, 21),
    "Total size (width × height) must be <= 2^21 pixels",
  );

export interface ZhipuImageProviderOptions {
  /**
   * The unique ID of the end user, helps the platform intervene in illegal activities, generate illegal or improper information, or other abuse by the end user.
   * ID length requirement: at least 6 characters, up to 128 characters.
   */
  userId?: string;
  /**
   * Override the quality setting for the image generation, defaults to "standard".
   * Choices are "standard" and "hd". Adjustable for "cogview-4-250304" only.
   */
  quality?: "standard" | "hd";
}

export { sizeSchema };
