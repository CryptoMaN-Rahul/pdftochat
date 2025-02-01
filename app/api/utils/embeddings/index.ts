import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

export function loadEmbeddingsModel() {
  return new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_API_KEY,
    modelName: "models/text-embedding-004", // Default Google embedding model
  });
}
