import { createJsonErrorResponseHandler } from "@ai-sdk/provider-utils";
import { z, ZodType } from "zod";

export const zhipuErrorDataSchema = z.object({
  error: z.object({
    message: z.string(),
    code: z.union([z.string(), z.number()]).nullish(),
  }),
});

export type ZhipuErrorData = z.infer<typeof zhipuErrorDataSchema>;

export const zhipuFailedResponseHandler = createJsonErrorResponseHandler({
  errorSchema: zhipuErrorDataSchema,
  errorToMessage: (data: ZhipuErrorData) => data.error.message,
});

export type ProviderErrorStructure<T> = {
  errorSchema: ZodType<T>;
  errorToMessage: (error: T) => string;
  isRetryable?: (response: Response, error?: T) => boolean;
};

export const defaultZhipuErrorStructure: ProviderErrorStructure<ZhipuErrorData> =
  {
    errorSchema: zhipuErrorDataSchema,
    errorToMessage: (data) => data.error.message,
  };
