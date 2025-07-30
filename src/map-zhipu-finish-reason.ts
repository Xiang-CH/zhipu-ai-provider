import { LanguageModelV2FinishReason } from "@ai-sdk/provider";

export function mapZhipuFinishReason(
  finishReason: string | null | undefined,
): LanguageModelV2FinishReason {
  switch (finishReason) {
    case "stop":
      return "stop";
    case "length":
      return "length";
    case "tool_calls":
      return "tool-calls";
    case "sensitive":
      return "content-filter";
    case "network_error":
      return "error";
    default:
      return "unknown";
  }
}
