import React from 'react';
import { Language } from '../types';

interface Props {
  lang: Language;
}

const ArchitectureDiagram: React.FC<Props> = ({ lang }) => {
  const t = {
    en: {
      caption: "Figure 1: The Transformer - Model Architecture (Source: Attention Is All You Need)"
    },
    zh: {
      caption: "图 1：Transformer 模型架构图 (来源: Attention Is All You Need)"
    }
  }[lang];

  return (
    <div className="flex flex-col items-center my-8">
      <div className="bg-white p-4 rounded-xl shadow-md border border-slate-200">
        <img 
          src="https://upload.wikimedia.org/wikipedia/commons/9/91/Transformer_architecture.png" 
          alt="Transformer Architecture" 
          className="max-h-[600px] max-w-full object-contain"
        />
      </div>
      <p className="text-xs text-slate-500 mt-3 italic font-serif">{t.caption}</p>
    </div>
  );
};

export default ArchitectureDiagram;