import { embed } from "ai";
import { getProvider, models, section } from "./_shared.mjs";

const provider = getProvider();

const result = await embed({
  model: provider.embedding(models.embedding, {
    dimensions: 256,
  }),
  value:
    "Zhipu AI provider examples let maintainers quickly sanity-check text, tools, embeddings, and images.",
});

section("Embedding");
console.log(`Vector length: ${result.embedding.length}`);
console.log(result.embedding.slice(0, 12));
