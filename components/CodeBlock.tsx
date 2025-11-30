import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

interface Props {
  code: string;
  language?: string;
  title?: string;
}

const CodeBlock: React.FC<Props> = ({ code, language = "Python", title = "PyTorch / Pseudo-code" }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-6 rounded-lg overflow-hidden border border-slate-700 shadow-lg">
      <div className="bg-slate-800 text-slate-300 px-4 py-2 text-xs font-mono border-b border-slate-700 flex justify-between items-center">
        <span className="font-bold">{title}</span>
        <div className="flex items-center gap-3">
          <span className="text-xs opacity-50 uppercase">{language}</span>
          <button 
            onClick={handleCopy}
            className="hover:text-white transition-colors focus:outline-none"
            title="Copy code"
          >
            {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
          </button>
        </div>
      </div>
      <pre className="bg-slate-900 text-blue-100 p-4 overflow-x-auto text-sm font-mono leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  );
};

export default CodeBlock;