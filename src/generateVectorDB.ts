import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from '@langchain/core/documents';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai'
import { Index, Pinecone, RecordMetadata } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import dotEnv from 'dotenv';

dotEnv.config();

const DATA_PATH = 'src/data';

async function loadDocuments() {
  const loader = new DirectoryLoader(DATA_PATH, {
    '.txt': (path) => new TextLoader(path),
  });
  const documents = await loader.load();
  return documents;
}

async function splitText(documents: Document[]) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 500,
  });

  const chunks = await splitter.splitDocuments(documents);
  console.log(
    `Split ${documents.length} documents into ${chunks.length} chunks.`
  );

  const sampleDocument = chunks[10];
  console.log('\n');
  console.log('-- splitter ---');
  console.log('pageContent:', sampleDocument.pageContent);
  console.log('metadata:', sampleDocument.metadata);
  console.log('-- !splitter ---');
  console.log('\n');

  return chunks;
}

async function generateDataStore() {
  const documents = await loadDocuments();
  const chunks = await splitText(documents);
  await saveToPineConeVectorDB(chunks);
}

async function saveToPineConeVectorDB(chunks: Document[]) {
  const indexName = process.env.PINECONE_INDEX;

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
      dimension: 768,
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

    await PineconeStore.fromDocuments(chunks, new GoogleGenerativeAIEmbeddings({
      modelName: 'text-embedding-004'
    }), {
      pineconeIndex,
      maxConcurrency: 5,
    });

    console.log(`Saved ${chunks.length} chunks to index ${indexName}.`);
  } else {
    console.log(`index not defined in env`);
  }
}

generateDataStore();
