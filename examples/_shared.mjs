import process from "node:process";
import { createZhipu, zai, zhipu } from "../dist/index.mjs";

function requiredEnv(name) {
  const value = process.env[name];

  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }

  return value;
}

export function getProvider() {
  const apiKey = requiredEnv("ZHIPU_API_KEY");
  const baseURL = process.env.ZHIPU_BASE_URL;
  const providerName = process.env.ZHIPU_PROVIDER;

  if (baseURL) {
    return createZhipu({ apiKey, baseURL });
  }

  if (providerName === "zai") {
    return zai;
  }

  return zhipu;
}

export const models = {
  text: process.env.ZHIPU_TEXT_MODEL ?? "glm-4.7-flash",
  reasoning: process.env.ZHIPU_REASONING_MODEL ?? "glm-4.7-flash",
  vision: process.env.ZHIPU_VISION_MODEL ?? "glm-4.6v-flash",
  embedding: process.env.ZHIPU_EMBEDDING_MODEL ?? "embedding-3",
  image: process.env.ZHIPU_IMAGE_MODEL ?? "glm-image",
};

export function section(title) {
  console.log(`\n=== ${title} ===`);
}

export function printWarnings(warnings) {
  if (warnings && warnings.length > 0) {
    section("Warnings");
    console.dir(warnings, { depth: null });
  }
}

export function printReasoning(reasoning) {
  if (reasoning && reasoning.length > 0) {
    section("Reasoning");
    console.dir(reasoning, { depth: null });
  }
}

export function printToolActivity({ toolCalls, toolResults }) {
  if (toolCalls && toolCalls.length > 0) {
    section("Tool Calls");
    console.dir(toolCalls, { depth: null });
  }

  if (toolResults && toolResults.length > 0) {
    section("Tool Results");
    console.dir(toolResults, { depth: null });
  }
}

export function printSources(sources) {
  if (sources && sources.length > 0) {
    section("Sources");
    console.dir(sources, { depth: null });
  }
}
