import { createJsonErrorResponseHandler } from "@ai-sdk/provider-utils";
import { z } from "zod";

export const zhipuErrorDataSchema = z.object({
  error: z.object({
    message: z.string(),
    code: z.union([z.string(), z.number()]).nullish(),
  }),
});

export type ZhipuErrorData = z.infer<typeof zhipuErrorDataSchema>;

export const zhipuFailedResponseHandler = createJsonErrorResponseHandler({
  errorSchema: zhipuErrorDataSchema,
  errorToMessage: (data) => data.error.message,
});
