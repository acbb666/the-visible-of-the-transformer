import React from 'react';

export type Language = 'en' | 'zh';

export interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correctAnswer: number; // Index of the correct option
  explanation: string;
}

export interface Section {
  id: string;
  title: string;
  content: React.ReactNode;
  difficulty?: 'basic' | 'intermediate' | 'advanced';
}

export interface Chapter {
  id: string;
  title: string;
  description: string;
  sections: Section[];
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  isError?: boolean;
}

export enum ViewState {
  HOME = 'HOME',
  CHAPTER = 'CHAPTER',
}