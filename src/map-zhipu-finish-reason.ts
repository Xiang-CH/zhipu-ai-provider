import { LanguageModelV3FinishReason } from "@ai-sdk/provider";

export function mapZhipuFinishReason(
  finishReason: string | null | undefined,
): LanguageModelV3FinishReason {
  switch (finishReason) {
    case "stop":
      return { unified: "stop", raw: finishReason };
    case "length":
      return { unified: "length", raw: finishReason };
    case "tool_calls":
      return { unified: "tool-calls", raw: finishReason };
    case "sensitive":
      return { unified: "content-filter", raw: finishReason };
    case "network_error":
      return { unified: "error", raw: finishReason };
    default:
      return { unified: "other", raw: finishReason ?? undefined };
  }
}
