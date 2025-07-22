// https://bigmodel.cn/dev/howuse/model
export type ZhipuChatModelId =
  // Language models
  | "glm-4-plus"
  | "glm-4-air-250414"
  | "glm-4-air"
  | "glm-4-airx"
  | "glm-4-long"
  | "glm-4-flash"
  | "glm-4-flash-250414"
  | "glm-4-flashx"
  // Vision/Video Models
  | "glm-4v-plus-0111"
  | "glm-4v-plus"
  | "glm-4v"
  | "glm-4v-flash"
  // Reasoning Models
  | "glm-z1-air"
  | "glm-z1-airx"
  | "glm-z1-flash"
  // Vision Reasoning Models
  | "glm-4.1v-thinking-flash"
  | "glm-4.1v-thinking-flashx"
  | (string & {});

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
}
