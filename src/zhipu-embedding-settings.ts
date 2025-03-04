// https://bigmodel.cn/dev/api/vector/embedding
export type ZhipuEmbeddingModelId =
  | "embedding-2"
  | "embedding-3"
  | (string & {});

export interface ZhipuEmbeddingSettings {
  /**
   * Override the embedding dimension, defaults to 2048.
   * 256, 512, 1024 or 2048 are recommended for embedding-3.
   */
  dimensions?: number;
}
