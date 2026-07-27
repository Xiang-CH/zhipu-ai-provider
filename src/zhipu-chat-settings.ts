// https://bigmodel.cn/dev/howuse/model
export type ZhipuChatModelId =
  // Language models
  | "glm-5.2"
  | "glm-5.1"
  | "glm-5"
  | "glm-5-turbo"
  | "glm-4.7"
  | "glm-4.7-flash"
  | "glm-4.7-flashx"
  | "glm-4.6"
  | "glm-4.6-flash"
  | "glm-4.6-flashx"
  | "glm-4.5"
  | "glm-4.5-flash"
  | "glm-4.5-flashx"
  | "glm-4.5-air"
  | "glm-4.5-airx"
  | "glm-4-flash-250414"
  | "glm-4-flashx-250414"
  // Multimodal Models
  | "glm-5v-turbo"
  | "glm-4.6v"
  | "glm-4.6v-flash"
  | "glm-4.6v-flashx"
  | "glm-4.1v-thinking-flash"
  | "glm-4.1v-thinking-flashx"
  | (string & {});

/**
 * Controls the amount of reasoning GLM-5.2+ applies to a request.
 *
 * @see https://docs.z.ai/guides/capabilities/thinking
 */
export type ZhipuReasoningEffort =
  | "max"
  | "xhigh"
  | "high"
  | "medium"
  | "low"
  | "minimal"
  | "none";

/**
 * Thinking mode configuration for GLM-4.5+ models.
 * Enables deep reasoning capabilities for complex tasks.
 */
export interface ZhipuThinkingConfig {
  /**
   * Enable or disable thinking mode.
   * - "enabled": Model will use deep reasoning before responding
   * - "disabled": Standard response without explicit reasoning
   */
  type: "enabled" | "disabled";
  /**
   * Whether to clear thinking content from previous turns.
   * When true, previous reasoning is not retained in context.
   * @default false
   */
  clearThinking?: boolean;
}

export interface ZhipuChatSettings {
  /**
   * The unique ID of the end user, helps the platform intervene in illegal activities, generate illegal or improper information, or other abuse by the end user.
   * ID length requirement: at least 6 characters, up to 128 characters.
   */
  userId?: string;
  /**
   * The unique ID of the request, passed by the user side, must be unique;
   * The platform will generate one by default if not provided by the user side.
   */
  requestId?: string;
  /**
   * When do_sample is true, sampling strategy is enabled, when do_sample is false, the sampling strategy temperature, top_p will not take effect
   */
  doSample?: boolean;
  /**
   * Enable thinking/reasoning mode for GLM-4.5+ models.
   * When enabled, the model will perform deep reasoning before responding,
   * which improves performance on complex tasks like coding and multi-step reasoning.
   *
   * @see https://docs.z.ai/guides/llm/glm-4.7
   */
  thinking?: ZhipuThinkingConfig;
  /**
   * Controls the reasoning effort for GLM-5.2+ models.
   *
   * This does not enable thinking automatically. Set `thinking.type` to
   * `"enabled"` when deep thinking is required.
   *
   * @see https://docs.z.ai/guides/capabilities/thinking
   */
  reasoningEffort?: ZhipuReasoningEffort;
}
