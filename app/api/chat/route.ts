import { NextRequest, NextResponse } from 'next/server';
import type { Message as VercelChatMessage } from 'ai';
import { createRAGChain } from '@/utils/ragChain';
import type { Document } from '@langchain/core/documents';
import { HumanMessage, AIMessage, ChatMessage } from '@langchain/core/messages';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { type MongoClient } from 'mongodb';
import { loadRetriever } from '../utils/vector_store';
import { loadEmbeddingsModel } from '../utils/embeddings';
import { HarmCategory, HarmBlockThreshold } from "@google/generative-ai";


export const runtime =
  process.env.NEXT_PUBLIC_VECTORSTORE === 'mongodb' ? 'nodejs' : 'edge';

const formatVercelMessages = (message: VercelChatMessage) => {
  if (message.role === 'user') {
    return new HumanMessage(message.content);
  } else if (message.role === 'assistant') {
    return new AIMessage(message.content);
  } else {
    console.warn(
      `Unknown message type passed: "${message.role}". Falling back to generic message type.`,
    );
    return new ChatMessage({ content: message.content, role: message.role });
  }
};

export async function POST(req: NextRequest) {
  let mongoDbClient: MongoClient | undefined;

  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    
    if (!messages.length) {
      throw new Error('No messages provided.');
    }

    const formattedPreviousMessages = messages
      .slice(0, -1)
      .map(formatVercelMessages);
    const currentMessageContent = messages[messages.length - 1].content;
    const chatId = body.chatId;

    // Initialize Google Gemini Pro 1.5 model
    const model = new ChatGoogleGenerativeAI({
      modelName: 'gemini-1.5-pro-latest',
      apiKey: process.env.GOOGLE_API_KEY,
      temperature: 0,
      safetySettings: [
        {
          category: HarmCategory.HARM_CATEGORY_HARASSMENT, // Use enum instead of string
          threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH, // Use enum instead of string
        },
        {
          category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
          threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
      ],
    });

    const embeddings = loadEmbeddingsModel();

    let resolveWithDocuments: (value: Document[]) => void;
    const documentPromise = new Promise<Document[]>((resolve) => {
      resolveWithDocuments = resolve;
    });

    const retrieverInfo = await loadRetriever({
      chatId,
      embeddings,
      callbacks: [
        {
          handleRetrieverEnd(documents) {
            resolveWithDocuments(documents);
          },
        },
      ],
    });

    const retriever = retrieverInfo.retriever;
    mongoDbClient = retrieverInfo.mongoDbClient;

    const ragChain = await createRAGChain(model, retriever);

    const stream = await ragChain.stream({
      input: currentMessageContent,
      chat_history: formattedPreviousMessages,
    });

    const documents = await documentPromise;
    const serializedSources = Buffer.from(
      JSON.stringify(
        documents.map((doc) => ({
          pageContent: doc.pageContent.slice(0, 50) + '...',
          metadata: doc.metadata,
        })),
      ),
    ).toString('base64');

    const byteStream = stream.pipeThrough(new TextEncoderStream());

    return new Response(byteStream, {
      headers: {
        'x-message-index': (formattedPreviousMessages.length + 1).toString(),
        'x-sources': serializedSources,
      },
    });
  } catch (e: any) {
    console.error('Error in RAG processing:', e);
    return NextResponse.json(
      { error: e.message || 'An error occurred during processing' },
      { status: 500 },
    );
  } finally {
    if (mongoDbClient) {
      await mongoDbClient.close();
    }
  }
}
