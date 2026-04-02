import { generateText } from "ai";
import {
  getProvider,
  models,
  printWarnings,
  section,
} from "./_shared.mjs";

const provider = getProvider();

const result = await generateText({
  model: provider(models.reasoning, {
    thinking: {
      type: "enabled",
    },
  }),
  prompt:
    "Reason step by step about how to test a custom AI SDK provider migration, then give a compact final checklist.",
});

section("Text");
console.log(result.text);
section("Reasoning");
console.log(result.reasoningText);
printWarnings(result.warnings);
