import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from '@langchain/core/documents';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import dotEnv from 'dotenv';
import { saveToChromaDB } from './vectordb';

dotEnv.config();

const DATA_PATH = 'documents';

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

  // const sampleDocument = chunks[10];
  // console.log('\n');
  // console.log('-- splitter ---');
  // console.log('pageContent:', sampleDocument.pageContent);
  // console.log('metadata:', sampleDocument.metadata);
  // console.log('-- !splitter ---');
  // console.log('\n');

  return chunks;
}

async function generateDataStore() {
  const documents = await loadDocuments();
  const chunks = await splitText(documents);
  await saveToChromaDB(chunks, {
    dimension: 768,
    embeddings: new GoogleGenerativeAIEmbeddings({
      model: 'text-embedding-004'
    }),
  }, {
    collectionName: 'simple-document-collection'
  });
}

generateDataStore();
