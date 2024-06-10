import { Chroma } from '@langchain/community/vectorstores/chroma';
import { ChromaClient } from 'chromadb';
import { Document } from '@langchain/core/documents';
import type { EmbeddingsInterface } from '@langchain/core/embeddings';

export default async function saveToVectorDB(
  chunks: Document[],
  config: {
    embeddings: EmbeddingsInterface;
    dimension: number;
  },
  chromaDbConfigs: {
    collectionName: string
  }
) {
  const { collectionName } = chromaDbConfigs;

  const chromaClient = new ChromaClient();

  const allCollections = await chromaClient.listCollections();

  if (allCollections.some((item) => item.name === collectionName)) {
    console.log(`collection ${collectionName} exits`);
    await chromaClient.deleteCollection({ name: collectionName });
    console.log(`deleted ${collectionName}`);
  }

  const vectorStore = await Chroma.fromDocuments(chunks, config.embeddings, {
    numDimensions: config.dimension,
    collectionName,
  });

  console.log(`Saved ${chunks.length} chunks to collection ${collectionName}.`);

  return vectorStore;
}
