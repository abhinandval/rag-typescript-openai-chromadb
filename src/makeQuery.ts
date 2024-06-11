import rl from 'readline';
import {
  GoogleGenerativeAIEmbeddings,
  ChatGoogleGenerativeAI,
} from '@langchain/google-genai';
import dotEnv from 'dotenv';
import { PromptTemplate } from '@langchain/core/prompts';
import { Chroma } from '@langchain/community/vectorstores/chroma';

dotEnv.config();

const PROMPT_TEMPLATE = `
<s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
maximum and keep the answer concise. [/INST] </s> 
[INST] Question: {question} 
Context: {context} 
Answer: [/INST]
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

  const chromaVectorStore = await Chroma.fromExistingCollection(
    new GoogleGenerativeAIEmbeddings({
      modelName: 'text-embedding-004',
    }),
    {
      collectionName: 'simple-document-collection',
      numDimensions: 768,
    }
  );

  //! Search DB
  const k = 3;
  const results = await chromaVectorStore.similaritySearchWithScore(
    query_text,
    k
  );


  const contextText = results
    .map(([doc, _score]) => doc.pageContent)
    .join('\n\n---\n\n');
  const promptTemplate = PromptTemplate.fromTemplate(PROMPT_TEMPLATE);
  const finalPrompt = await promptTemplate.format({
    context: contextText,
    question: query_text,
  });
  console.log('finalPrompt:', finalPrompt);

  const model = new ChatGoogleGenerativeAI({
    model: 'gemini-1.5-pro-latest'
  });
  const response = await model.invoke(finalPrompt);

  const sources = results.map(
    ([doc]) => (doc.metadata['source'] as string) ?? 'no-source'
  );
  const formattedResponse = `\nResponse: ${response.content}\n\nSources: ${sources}`;
  console.log(formattedResponse);
}

main();
