# Zhipu AI Provider - Vercel AI SDK Community Provider

This is a [Zhipu](https://www.zhipuai.cn/) (Z.ai) provider for the [Vercel AI SDK](https://ai-sdk.dev/). It supports AI SDK 6 and the `LanguageModelV3` custom-provider contract for language models, plus embedding and image models provided on [bigmodel.cn](https://bigmodel.cn/) or [z.ai](https://docs.z.ai/) by [ZhipuAI](https://www.zhipuai.cn/).

## Setup

```bash
# npm
npm i zhipu-ai-provider

# pnpm
pnpm add zhipu-ai-provider

# yarn
yarn add zhipu-ai-provider

# bun
bun add zhipu-ai-provider
```

Set up your `.env` file / environment with your API key.

```bash
ZHIPU_API_KEY=<your-api-key>
```

## Provider Instance

You can import the default provider instance `zhipu` from `zhipu-ai-provider` (This automatically reads the API key from the environment variable `ZHIPU_API_KEY`):

```ts
import { zhipu } from 'zhipu-ai-provider' // for bigmodel.cn
// or
import { zai } from 'zhipu-ai-provider' // for z.ai
```

Alternatively, you can create a provider instance with custom configuration with `createZhipu`:

```ts
import { createZhipu } from 'zhipu-ai-provider';

const zhipu = createZhipu({
  baseURL: "https://open.bigmodel.cn/api/paas/v4",
  apiKey: "your-api-key"
});
```

You can use the following optional settings to customize the Zhipu provider instance:

- **baseURL**: *string*
  - Use a different URL prefix for API calls, e.g. to use proxy servers. The default prefix is `https://open.bigmodel.cn/api/paas/v4`.
- **apiKey**: *string*
  - Your API key for Zhipu [BigModel Platform](https://bigmodel.cn/). If not provided, the provider will attempt to read the API key from the environment variable `ZHIPU_API_KEY`.
- **headers**: *Record<string,string>*
  - Custom headers to include in the requests.

## Language Model Example

```ts
import { generateText } from 'ai';
import { zhipu } from 'zhipu-ai-provider';

const { text } = await generateText({
  model: zhipu('glm-5'), // or use 'GLM-4.7-Flash' for free
  prompt: 'Why is the sky blue?',
});

console.log(result)
```

To disable thinking for hybrid models like `glm-5`, set `thinking.type` to `disabled` either in the model options or in `providerOptions.zhipu`:

```ts
const { text } = await generateText({
  model: zhipu('glm-5', {
    thinking: {
      type: 'disabled'
    },
  }),
  prompt: 'Explain quantum computing in simple terms.',
});
```

or

```ts
const { text } = await generateText({
  model: zhipu('glm-5'),
  prompt: 'Explain quantum computing in simple terms.',
  providerOptions: {
    zhipu: {
      thinking: {
        type: 'disabled'
      }
    }
  }
});
```

Only function tools are supported. Provider-defined tools are not currently implemented.

## Embedding Example

```ts
const { embedding } = await embed({
  model: zhipu.embeddingModel("embedding-3", {
    dimensions: 256, // Optional, defaults to 2048
  }),
  value: "Hello, world!",
});

console.log(embedding);
```

`textEmbeddingModel(...)` is still available as a deprecated compatibility alias.

## Image Generation Example

Zhipu supports image generation with `glm-image` or `cogview` models, but the api does not return images in base64 or buffer format, so the image urls are returned in the `providerMetadata` field.

```ts
import { experimental_generateImage as generateImage } from 'ai';
import { zhipu } from 'zhipu-ai-provider';

const { image, providerMetadata } = await generateImage({
  model: zhipu.imageModel('glm-image'),  // or use 'Cogview-3-Flash' for free
  prompt: 'A beautiful landscape with mountains and a river',
  size: '1024x1024',  // optional
  providerOptions: {  // optional
      zhipu: {
          quality: 'hd'
      }
  }
});

console.log(providerMetadata.zhipu.images[0].url)
```

## Features Support


| Feature                                                         | Zhipu ([bigmodel.cn](https://bigmodel.cn/)) | Z.ai ([z.ai](https://z.ai/)) |
| --------------------------------------------------------------- | ------------------------------------------- | ---------------------------- |
| Text generation                                                 | ✓                                           | ✓                            |
| Streaming                                                       | ✓                                           | ✓                            |
| Embedding                                                       | ✓                                           | x                            |
| Image generation                                                | ✓                                           | ✓                            |
| Tools                                                           | ✓                                           | ✓                            |
| JSON response format                                            | ✓                                           | ✓                            |
| Reasoning                                                       | ✓                                           | ✓                            |
| Vision                                                          | ✓                                           | ✓                            |
| Vision reasoning                                                | ✓                                           | ✓                            |
| Schema-guided structured output for reasoning and vision models | —                                           | —                            |
| Provider-defined tools                                          | —                                           | —                            |
| Video models                                                    | —                                           | —                            |
| Audio models                                                    | —                                           | —                            |
| OCR models                                                      | —                                           | —                            |


## Documentation

- **[Zhipu documentation](https://bigmodel.cn/dev/welcome)** 
- **[Z.AI documentation](https://docs.z.ai/)**
- **[Vercel AI SDK documentation](https://sdk.vercel.ai/docs/introduction)**
- **[Zhipu AI Provider Repo](https://github.com/Xiang-CH/zhipu-ai-provider)**

## Maintainer Examples

Runnable maintainer-facing demo scripts live in [examples/README.md](/Users/cxiang/Projects/vercel-ai-provider-zhipu/examples/README.md). They cover text generation, streaming, reasoning, tool calls, vision prompts, embeddings, and image generation.