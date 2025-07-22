import {
  LanguageModelV2,
  LanguageModelV2CallWarning,
  UnsupportedFunctionalityError,
} from "ai";

export function prepareTools(
  mode: Parameters<LanguageModelV2["doGenerate"]>[0]["mode"] & {
    type: "regular";
  },
): {
  tools:
    | Array<{
        type: "function";
        function: {
          name: string;
          description: string | undefined;
          parameters: unknown;
        };
      }>
    | undefined;
  tool_choice: "auto" | undefined;
  toolWarnings: LanguageModelV2CallWarning[];
} {
  // when the tools array is empty, change it to undefined to prevent errors:
  const tools = mode.tools?.length ? mode.tools : undefined;
  const toolWarnings: LanguageModelV2CallWarning[] = [];

  if (tools == null) {
    return { tools: undefined, tool_choice: undefined, toolWarnings };
  }

  const zhipuTools: Array<{
    type: "function";
    function: {
      name: string;
      description: string | undefined;
      parameters: unknown;
    };
  }> = [];

  for (const tool of tools) {
    if (tool.type === "provider-defined") {
      toolWarnings.push({ type: "unsupported-tool", tool });
    } else {
      zhipuTools.push({
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      });
    }
  }

  const toolChoice = mode.toolChoice;

  if (toolChoice == null) {
    return { tools: zhipuTools, tool_choice: undefined, toolWarnings };
  }

  const type = toolChoice.type;

  switch (type) {
    case "none":
      return { tools: zhipuTools, tool_choice: undefined, toolWarnings };
    case "auto":
    case "required":
      return { tools: zhipuTools, tool_choice: "auto", toolWarnings };

    // zhipu does not support tool mode directly,
    // so we filter the tools and force the tool choice through 'any'
    case "tool":
      return {
        tools: zhipuTools.filter(
          (tool) => tool.function.name === toolChoice.toolName,
        ),
        tool_choice: "auto",
        toolWarnings,
      };
    default: {
      const _exhaustiveCheck: never = type;
      throw new UnsupportedFunctionalityError({
        functionality: `Unsupported tool choice type: ${_exhaustiveCheck}`,
      });
    }
  }
}
