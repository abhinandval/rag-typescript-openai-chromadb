import { Index, Pinecone, RecordMetadata } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import type { EmbeddingsInterface } from '@langchain/core/embeddings';
import { Document } from '@langchain/core/documents';

export default async function saveToVectorDB(
  chunks: Document[],
  config: {
    embeddings: EmbeddingsInterface;
    dimension: number;
  },
  pineconeConfig: {
    indexName: string;
  }
) {
  const { dimension, embeddings } = config;
  const { indexName } = pineconeConfig;

  if (indexName) {
    const pinecone = new Pinecone();
    const allIndices = await pinecone.listIndexes();
    if (allIndices.indexes?.some((item) => item.name === indexName)) {
      console.log(`Index ${indexName} exits`);

      await pinecone.deleteIndex(indexName);

      console.log(`Deleted index ${indexName}`);
    }

    await pinecone.createIndex({
      name: indexName,
      dimension: dimension,
      metric: 'cosine',
      spec: {
        serverless: {
          cloud: 'aws',
          region: 'us-east-1',
        },
      },
    });

    console.log(`Created new index with name ${indexName}`);

    const pineconeIndex = pinecone.Index(indexName);

    await PineconeStore.fromDocuments(chunks, embeddings, {
      pineconeIndex,
      maxConcurrency: 5,
    });

    console.log(`Saved ${chunks.length} chunks to index ${indexName}.`);
  } else {
    console.log(`index not defined in env`);
  }
}
