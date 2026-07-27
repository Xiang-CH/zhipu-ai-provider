import { streamText } from "ai";
import {
  getProvider,
  models,
  printReasoning,
  printWarnings,
  section,
} from "./_shared.mjs";

const provider = getProvider();

const result = streamText({
  model: provider.chat(models.text),
  prompt: "Write a short release note for migrating a custom AI SDK provider to LanguageModelV4.",
});

section("Stream");
for await (const delta of result.textStream) {
  process.stdout.write(delta);
}
process.stdout.write("\n");

printReasoning(await result.reasoning);
printWarnings(await result.warnings);
