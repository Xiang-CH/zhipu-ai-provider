import { generateImage } from "ai";
import { getProvider, models, section } from "./_shared.mjs";

const provider = getProvider();

const result = await generateImage({
  model: provider.imageModel(models.image),
  prompt:
    "A crisp product illustration of a green terminal window testing an AI SDK provider, minimal poster style.",
  size: "1024x1024",
});

section("Images");
console.log(`Returned images: ${result.images.length}`);

section("Provider Metadata");
console.dir(result.providerMetadata, { depth: null });
