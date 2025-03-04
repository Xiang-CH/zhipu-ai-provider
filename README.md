# Zhipu AI Provider - Vercel AI SDK Community Provider
This is a [Zhipu](https://www.zhipuai.cn/) prodiver for the [Vercel AI](https://sdk.vercel.ai/) SDK. It enables seamless integration with **GLM** and Embedding Models provided on [bigmodel.cn](https://bigmodel.cn/).


## Setup

```bash
npm i zhipu-ai-provider
export ZHIPU_API_KEY=<your-api-key>
```


```ts
import { zhipu } from 'zhipu-ai-provider'
```
or
```ts
import { createZhipu } from 'zhipu-ai-provider';

const zhipu = createZhipu({
  baseURL: "https://open.bigmodel.cn/api/paas/v4",
  apiKey: "your-api-key"
});
```

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
Please check out the **[Zhipu documentation](https://bigmodel.cn/dev/welcome)** for more information.
