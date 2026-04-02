# Examples

These examples are intended for maintainers to quickly exercise the provider against live APIs.

## Setup

1. Build once so the examples can import the local package entrypoint:

```bash
pnpm build
```

2. Export your API key:

```bash
export ZHIPU_API_KEY=your-api-key
```

Optional overrides:

```bash
export ZHIPU_PROVIDER=zai
export ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
export ZHIPU_TEXT_MODEL=glm-4-flash
export ZHIPU_REASONING_MODEL=glm-4.5
export ZHIPU_VISION_MODEL=glm-4.1v-thinking-flash
export ZHIPU_EMBEDDING_MODEL=embedding-3
export ZHIPU_IMAGE_MODEL=glm-image
export EXAMPLE_IMAGE_URL=https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg
```

## Run

```bash
pnpm example:generate-text
pnpm example:stream-text
pnpm example:generate-text-reasoning
pnpm example:generate-text-tool-call
pnpm example:generate-text-with-images
pnpm example:embed
pnpm example:generate-image
```

## Coverage

- `generate-text.mjs`: basic text generation
- `stream-text.mjs`: streamed text output
- `generate-text-reasoning.mjs`: reasoning-enabled text generation without tools
- `generate-text-tool-call.mjs`: reasoning plus function tool calls
- `generate-text-with-images.mjs`: multimodal prompt with an image URL
- `embed.mjs`: text embeddings
- `generate-image.mjs`: image generation
