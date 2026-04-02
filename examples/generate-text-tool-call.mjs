import { generateText, stepCountIs, tool } from "ai";
import { z } from "zod";
import {
  getProvider,
  models,
  printReasoning,
  printToolActivity,
  printWarnings,
  section,
} from "./_shared.mjs";

const provider = getProvider();

const result = await generateText({
  model: provider.chat(models.reasoning, {
    thinking: {
      type: "enabled",
    },
  }),
  prompt:
    "Use the weather tool to compare Hong Kong and San Francisco, then recommend which city is better for a demo day.",
  tools: {
    weather: tool({
      description: "Returns mocked weather for a city so the example can verify tool calling.",
      inputSchema: z.object({
        city: z.string(),
      }),
      execute: async ({ city }) => {
        const mockWeather = {
          "hong kong": { condition: "humid", temperatureC: 27 },
          "san francisco": { condition: "cool", temperatureC: 18 },
        };

        return mockWeather[city.toLowerCase()] ?? {
          condition: "unknown",
          temperatureC: 22,
        };
      },
    }),
  },
  stopWhen: stepCountIs(3),
});

section("Text");
console.log(result.text);
section("Reasoning");
console.log(result.reasoningText);
printToolActivity({
  toolCalls: result.toolCalls,
  toolResults: result.toolResults,
});
printWarnings(result.warnings);
