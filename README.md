# Zhipu AI Provider - Vercel AI SDK Community Provider
This is a [Zhipu](https://www.zhipuai.cn/) prodiver for the [Vercel AI](https://sdk.vercel.ai/) SDK. It enables seamless integration with **GLM** and Embedding Models provided on [bigmodel.cn](https://bigmodel.cn/).


## Setup

```bash
# npm
npm i zhipu-ai-provider

# pnpm
pnpm add zhipu-ai-provider

# yarn
yarn add zhipu-ai-provider
```
Set up your `.env` file / environment with your API key.
```bash
ZHIPU_API_KEY=<your-api-key>
```

## Provider Instance
You can import the default provider instance `zhipu` from `zhipu-ai-provider` (This automatically reads the API key from the environment variable `ZHIPU_API_KEY`):
```ts
import { zhipu } from 'zhipu-ai-provider'
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

## Example

```ts
import { zhipu } from 'zhipu-ai-provider';

const { text } = await generateText({
  model: zhipu('glm-4-plus'),
  prompt: 'Why is the sky blue?',
});

console.log(result)
```

## Documentation
- **[Zhipu documentation](https://bigmodel.cn/dev/welcome)** 
- **[Vercel AI SDK documentation](https://sdk.vercel.ai/docs/introduction)**