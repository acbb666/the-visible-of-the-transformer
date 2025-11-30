import { GoogleGenAI } from "@google/genai";
import { Language } from "../types";

const API_KEY = process.env.API_KEY || '';

const ai = new GoogleGenAI({ apiKey: API_KEY });

export const askTutor = async (question: string, lang: Language = 'en', history: string[] = []): Promise<string> => {
  if (!API_KEY) {
    return lang === 'zh' 
      ? "演示模式：缺少 API 密钥。请配置 process.env.API_KEY 以使用 AI 导师。"
      : "Demo Mode: API Key is missing. Please configure process.env.API_KEY to use the AI Tutor.";
  }

  try {
    const model = 'gemini-2.5-flash';
    const langInstruction = lang === 'zh' 
      ? 'Answer in Chinese (Simplified). Use professional terminology suitable for machine learning context.' 
      : 'Answer in English.';

    const systemInstruction = `You are a specialized AI Tutor for the "Transformer" deep learning architecture and the paper "Attention Is All You Need". 
    Your goal is to explain complex concepts (Self-Attention, Multi-Head Attention, Encoder-Decoder, etc.) in simple, intuitive terms.
    Use analogies where appropriate. Keep answers concise but accurate.
    If the user asks about code, provide PyTorch-style pseudocode.
    Do not answer questions unrelated to Machine Learning or Transformers.
    ${langInstruction}`;

    const response = await ai.models.generateContent({
      model,
      contents: question,
      config: {
        systemInstruction,
        temperature: 0.7,
      }
    });

    return response.text || (lang === 'zh' ? "无法生成回复，请重试。" : "I couldn't generate a response. Please try again.");
  } catch (error) {
    console.error("Gemini API Error:", error);
    return lang === 'zh' 
      ? "抱歉，连接知识库时出现错误。" 
      : "Sorry, I encountered an error connecting to the knowledge base.";
  }
};