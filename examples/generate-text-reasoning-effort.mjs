import process from "node:process";
import { generateText } from "ai";
import { getProvider, printWarnings, section } from "./_shared.mjs";

const provider = getProvider();
const modelId = process.env.ZHIPU_REASONING_EFFORT_MODEL ?? "glm-5.2";

const result = await generateText({
  model: provider(modelId),
  prompt:
    "Design a resilient recommendation system, then summarize the tradeoffs in a compact checklist.",
  providerOptions: {
    zhipu: {
      thinking: { type: "enabled" },
      reasoningEffort: "high",
    },
  },
});

section("Text");
console.log(result.text);
section("Reasoning");
console.log(result.reasoningText);
printWarnings(result.warnings);
