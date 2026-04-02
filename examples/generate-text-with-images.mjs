import { generateText } from "ai";
import {
  getProvider,
  models,
  printReasoning,
  printWarnings,
  section,
} from "./_shared.mjs";

const provider = getProvider();
const imageUrl =
  process.env.EXAMPLE_IMAGE_URL ??
  "https://img.notionusercontent.com/ext/https%3A%2F%2Fcdn.cxiang.site%2Fblog_covers%2Flatex.webp/size/?exp=1775150636&sig=rsDi4cFLVQ8rqDGjUwys1HPQzwZDyPQE1PH9bMgDU2M&id=293f94f7-98c1-8070-aa79-e92e19614d3c&table=block&userId=f0e00391-0fe2-4fb6-8c28-3680117d1d29";

const result = await generateText({
  model: provider(models.vision),
  messages: [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: "Describe the image in 3 short bullet points and mention the likely setting.",
        },
        {
          type: "image",
          image: new URL(imageUrl),
        },
      ],
    },
  ],
  providerOptions: {
    zhipu: {
      reasoning: {
        type: "enabled",
      },
    },
  },
});

section("Text");
console.log(result.text);
printReasoning(result.reasoning);
printWarnings(result.warnings);
