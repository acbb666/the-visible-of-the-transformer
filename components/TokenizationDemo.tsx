import React, { useState } from 'react';
import { Language } from '../types';

interface Props {
  lang: Language;
}

const TokenizationDemo: React.FC<Props> = ({ lang }) => {
  const [text, setText] = useState(
    lang === 'zh' ? "Transformer 模型改变了自然语言处理。" : "The Transformer model changed NLP forever."
  );

  const t = {
    en: {
      title: "Interactive: From Text to Tokens",
      input: "Type a sentence:",
      output: "Tokens (Simulated Sub-word / BPE):",
      desc: "Notice how common words are kept whole, but rarer words or suffixes might be split. The model sees these integer IDs, not the text.",
      ids: "Token IDs"
    },
    zh: {
      title: "交互演示：从文本到 Token",
      input: "输入一个句子：",
      output: "Token 序列 (模拟子词/BPE 分词)：",
      desc: "注意：常用词通常保持完整，而较生僻的词或词缀可能会被拆分。模型看到的是这些 Token 对应的整数 ID，而不是文本。",
      ids: "Token ID"
    }
  }[lang];

  // A very naive simulation of BPE/Subword tokenization for visual purposes
  const simulateTokenization = (input: string) => {
    const tokens: string[] = [];
    const words = input.split(' ');
    
    words.forEach(word => {
      if (word.length > 5 && !['Transformer', 'change', 'model'].includes(word.replace(/[.,]/g, ''))) {
        // Split long words arbitrarily to simulate sub-words
        const mid = Math.floor(word.length / 2);
        tokens.push(word.slice(0, mid));
        tokens.push("##" + word.slice(mid));
      } else {
        tokens.push(word);
      }
    });
    return tokens;
  };

  const tokens = simulateTokenization(text);
  const colors = [
    "bg-blue-100 text-blue-800 border-blue-200",
    "bg-green-100 text-green-800 border-green-200",
    "bg-purple-100 text-purple-800 border-purple-200",
    "bg-orange-100 text-orange-800 border-orange-200",
    "bg-pink-100 text-pink-800 border-pink-200",
  ];

  return (
    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm my-6">
      <h3 className="font-bold text-slate-800 mb-4">{t.title}</h3>
      
      <div className="mb-4">
        <label className="block text-sm font-medium text-slate-500 mb-2">{t.input}</label>
        <input 
          type="text" 
          value={text} 
          onChange={(e) => setText(e.target.value)}
          className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-brand-500 focus:border-transparent outline-none transition"
        />
      </div>

      <div className="mb-2">
        <label className="block text-sm font-medium text-slate-500 mb-2">{t.output}</label>
        <div className="flex flex-wrap gap-2 p-4 bg-slate-50 rounded-lg border border-slate-100 min-h-[60px]">
          {tokens.map((token, i) => (
            <div key={i} className="flex flex-col items-center group cursor-default">
              <span className={`px-3 py-1 rounded-md border text-sm font-mono font-bold shadow-sm ${colors[i % colors.length]}`}>
                {token}
              </span>
              <span className="text-[10px] text-slate-400 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
                {1000 + i * 23}
              </span>
            </div>
          ))}
        </div>
      </div>
      
      <p className="text-xs text-slate-500 mt-2">{t.desc}</p>
    </div>
  );
};

export default TokenizationDemo;