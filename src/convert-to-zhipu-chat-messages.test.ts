import { describe, expect, it } from "vitest";
import { convertToZhipuChatMessages } from "./convert-to-zhipu-chat-messages";

describe("user messages", () => {
  it("should convert messages with image parts", () => {
    const result = convertToZhipuChatMessages([
      {
        role: "user",
        content: [
          { type: "text", text: "Hello" },
          {
            type: "file",
            data: { type: "data", data: new Uint8Array([0, 1, 2, 3]) },
            mediaType: "image/png",
          },
        ],
      },
    ]);

    expect(result).toMatchSnapshot();
  });
});

describe("tool calls", () => {
  it("should stringify arguments to tool calls", () => {
    const result = convertToZhipuChatMessages([
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            input: { key: "arg-value" },
            toolCallId: "tool-call-id-1",
            toolName: "tool-1",
          },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "tool-call-id-1",
            toolName: "tool-1",
            output: {
              type: "json",
              value: { key: "result-value" },
            },
          },
        ],
      },
    ]);

    expect(result).toMatchSnapshot();
  });
});

describe("assistant messages", () => {
  it("should add prefix true to trailing assistant messages", () => {
    const result = convertToZhipuChatMessages([
      {
        role: "user",
        content: [{ type: "text", text: "Hello" }],
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "Hello!" }],
      },
    ]);

    expect(result).toMatchSnapshot();
  });

  it("should reject unsupported V4 assistant content parts", () => {
    expect(() =>
      convertToZhipuChatMessages([
        {
          role: "assistant",
          content: [
            {
              type: "custom",
              kind: "zhipu.thinking",
            },
          ],
        },
      ]),
    ).toThrow("Assistant custom content parts");
  });
});

describe("tool messages", () => {
  it("should reject unsupported V4 tool approval responses", () => {
    expect(() =>
      convertToZhipuChatMessages([
        {
          role: "tool",
          content: [
            {
              type: "tool-approval-response",
              approvalId: "approval-1",
              approved: true,
            },
          ],
        },
      ]),
    ).toThrow("Tool approval response content parts");
  });
});
