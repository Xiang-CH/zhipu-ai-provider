import type { JSONValue } from "ai";

export type ZhipuPrompt = Array<ZhipuMessage>;

export type ZhipuMessage =
  | ZhipuSystemMessage
  | ZhipuUserMessage
  | ZhipuAssistantMessage
  | ZhipuToolMessage;

type JsonRecord<T = never> = Record<
  string,
  JSONValue | JSONValue[] | T | T[] | undefined
>;

export interface ZhipuSystemMessage extends JsonRecord {
  role: "system";
  content: string;
}

export interface ZhipuUserMessage extends JsonRecord<ZhipuUserMessageParts> {
  role: "user";
  content: string | Array<ZhipuUserMessageParts>;
}

export type ZhipuUserMessageParts =
  | ZhipuUserMessageTextPart
  | ZhipuUserMessageImagePart
  | ZhipuUserMessageVideoPart;

export interface ZhipuUserMessageTextPart extends JsonRecord {
  type: "text";
  text: string;
}

export interface ZhipuUserMessageImagePart extends JsonRecord {
  type: "image_url";
  image_url: {
    url: string;
  };
}

export interface ZhipuUserMessageVideoPart extends JsonRecord {
  type: "video_url";
  video_url: {
    url: string;
  };
}

export interface ZhipuAssistantMessage extends JsonRecord {
  role: "assistant";
  content: string;
  prefix?: boolean;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
}

export interface ZhipuToolMessage extends JsonRecord {
  role: "tool";
  content: string;
  tool_call_id: string;
}
