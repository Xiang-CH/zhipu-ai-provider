import {
  LanguageModelV4Prompt,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";
import { convertUint8ArrayToBase64 } from "@ai-sdk/provider-utils";
import { ZhipuPrompt } from "./zhipu-chat-prompt";

export function convertToZhipuChatMessages(
  prompt: LanguageModelV4Prompt,
): ZhipuPrompt {
  const messages: ZhipuPrompt = [];

  for (let i = 0; i < prompt.length; i++) {
    const { role, content } = prompt[i];
    const isLastMessage = i === prompt.length - 1;

    switch (role) {
      case "system": {
        messages.push({ role: "system", content });
        break;
      }

      case "user": {
        if (content.length === 1 && content[0].type === "text") {
          messages.push({ role: "user", content: content[0].text });
          break;
        }

        if (content.every((part) => part.type === "text")) {
          messages.push({
            role: "user",
            content: content.map((part) => part.text).join(""),
          });
          break;
        }

        messages.push({
          role: "user",
          content: content.map((part) => {
            switch (part.type) {
              case "text": {
                return { type: "text", text: part.text };
              }
              case "file": {
                if (
                  part.mediaType.startsWith("image/") &&
                  (part.data.type === "data" || part.data.type === "url")
                ) {
                  return {
                    type: "image_url",
                    image_url: {
                      url:
                        part.data.type === "url"
                          ? part.data.url.toString()
                          : typeof part.data.data === "string"
                            ? `data:${part.mediaType};base64,${part.data.data}`
                            : `data:${part.mediaType};base64,${convertUint8ArrayToBase64(part.data.data)}`,
                    },
                  };
                }

                if (
                  part.mediaType.startsWith("video/") &&
                  part.data.type === "url"
                ) {
                  return {
                    type: "video_url",
                    video_url: {
                      url: part.data.url.toString(),
                    },
                  };
                }

                throw new UnsupportedFunctionalityError({
                  functionality: "File content parts in user messages",
                });
              }
            }
          }),
        });
        break;
      }

      case "assistant": {
        let text = "";
        const toolCalls: Array<{
          id: string;
          type: "function";
          function: { name: string; arguments: string };
        }> = [];

        for (const part of content) {
          switch (part.type) {
            case "text": {
              text += part.text;
              break;
            }
            case "reasoning":
            case "file":
            case "tool-result": {
              break;
            }
            case "reasoning-file":
            case "custom": {
              throw new UnsupportedFunctionalityError({
                functionality: `Assistant ${part.type} content parts`,
              });
            }
            case "tool-call": {
              toolCalls.push({
                id: part.toolCallId,
                type: "function",
                function: {
                  name: part.toolName,
                  arguments: JSON.stringify(part.input),
                },
              });
              break;
            }
            default: {
              const _exhaustiveCheck: never = part;
              throw new Error(`Unsupported part: ${_exhaustiveCheck}`);
            }
          }
        }

        messages.push({
          role: "assistant",
          content: text,
          prefix: isLastMessage ? true : undefined,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        });
        break;
      }

      case "tool": {
        for (const toolResponse of content) {
          if (toolResponse.type === "tool-approval-response") {
            throw new UnsupportedFunctionalityError({
              functionality: "Tool approval response content parts",
            });
          }

          const output = toolResponse.output;
          let contentValue: string;

          switch (output.type) {
            case "text":
            case "error-text":
              contentValue = output.value;
              break;
            case "execution-denied":
              contentValue = output.reason ?? "Tool execution denied";
              break;
            case "content":
            case "json":
            case "error-json":
              contentValue = JSON.stringify(output.value);
              break;
            default: {
              const _exhaustiveCheck: never = output;
              throw new Error(`Unsupported tool output: ${_exhaustiveCheck}`);
            }
          }

          messages.push({
            role: "tool",
            content: contentValue,
            tool_call_id: toolResponse.toolCallId,
          });
        }
        break;
      }

      default: {
        const _exhaustiveCheck: never = role;
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`);
      }
    }
  }

  return messages;
}
