import rl from 'readline';
import {
  GoogleGenerativeAIEmbeddings,
  ChatGoogleGenerativeAI,
} from '@langchain/google-genai';
import dotEnv from 'dotenv';
import { Index, Pinecone, RecordMetadata } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { PromptTemplate } from '@langchain/core/prompts';

dotEnv.config();

const PROMPT_TEMPLATE = `
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
`;

async function main() {
  const readline = rl.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const queryInput = new Promise<string | undefined>((resolve) => {
    readline.question('Enter your query: ', (answer) => {
      resolve(answer);
      readline.close();
    });
  });

  const query_text = await queryInput;

  if (!query_text) {
    console.error('Query is required');
    return;
  }

  //! Prepare DB
  const indexName = process.env.PINECONE_INDEX;
  if (!indexName) {
    console.log(`index not defined in env`);
    return;
  }

  const pinecone = new Pinecone();
  let pineconeIndex: Index<RecordMetadata>;

  //* Check if index exits
  const allIndices = await pinecone.listIndexes();
  if (allIndices.indexes?.some((item) => item.name === indexName)) {
    console.log(`Index ${indexName} exits`);
    pineconeIndex = pinecone.Index(indexName);
  } else {
    return;
  }

  const vectorStore = await PineconeStore.fromExistingIndex(
    new GoogleGenerativeAIEmbeddings({
      modelName: 'text-embedding-004',
    }),
    { pineconeIndex }
  );

  //! Search DB
  const k = 3;
  const results = await vectorStore.similaritySearchWithScore(query_text, k);

  if (results.length === 0 || results[0][1] < 0.6) {
    console.error('Unable to find matching results');
    return;
  }

  const contextText = results
    .map(([doc, _score]) => doc.pageContent)
    .join('\n\n---\n\n');
  const promptTemplate = PromptTemplate.fromTemplate(PROMPT_TEMPLATE);
  const finalPrompt = await promptTemplate.format({
    context: contextText,
    question: query_text,
  });
  console.log('finalPrompt:', finalPrompt);

  const model = new ChatGoogleGenerativeAI();
  const response = await model.invoke(finalPrompt);

  const sources = results.map(
    ([doc]) => (doc.metadata['source'] as string) ?? 'no-source'
  );
  const formattedResponse = `\nResponse: ${(await response).content}\n\nSources: ${sources}`;
  console.log(formattedResponse);
}

main();
