import { generateText } from "ai";
import {
  getProvider,
  models,
  printReasoning,
  printSources,
  printWarnings,
  section,
} from "./_shared.mjs";

const provider = getProvider();

const result = await generateText({
  model: provider(models.text),
  prompt: "Explain in 3 concise bullet points what this Zhipu provider repository does.",
});

section("Text");
console.log(result.text);
printReasoning(result.reasoning);
printSources(result.sources);
printWarnings(result.warnings);
