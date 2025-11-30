import React, { useState } from 'react';
import { CheckCircle, XCircle, HelpCircle } from 'lucide-react';
import { QuizQuestion, Language } from '../types';

interface Props {
  question: QuizQuestion;
  lang: Language;
}

const QuizSection: React.FC<Props> = ({ question, lang }) => {
  const [selected, setSelected] = useState<number | null>(null);
  const [submitted, setSubmitted] = useState(false);

  const isCorrect = selected === question.correctAnswer;

  const t = {
    en: {
      check: "Check Answer",
      correct: "Correct!",
      incorrect: "Incorrect",
      retry: "Try Again",
      why: "Explanation:"
    },
    zh: {
      check: "提交答案",
      correct: "回答正确！",
      incorrect: "回答错误",
      retry: "重试",
      why: "解析："
    }
  }[lang];

  const handleSubmit = () => {
    if (selected !== null) setSubmitted(true);
  };

  const handleRetry = () => {
    setSelected(null);
    setSubmitted(false);
  };

  return (
    <div className="my-8 p-6 bg-white rounded-xl border-2 border-slate-100 shadow-sm">
      <div className="flex items-center gap-2 mb-4 text-brand-600 font-bold uppercase tracking-wider text-xs">
        <HelpCircle size={16} />
        <span>{lang === 'en' ? 'Quick Quiz' : '小测验'}</span>
      </div>
      
      <h3 className="text-lg font-bold text-slate-900 mb-4">{question.question}</h3>
      
      <div className="space-y-3">
        {question.options.map((option, idx) => (
          <button
            key={idx}
            onClick={() => !submitted && setSelected(idx)}
            disabled={submitted}
            className={`w-full text-left p-3 rounded-lg border transition-all relative ${
              submitted
                ? idx === question.correctAnswer
                  ? 'bg-green-50 border-green-500 text-green-900'
                  : idx === selected
                    ? 'bg-red-50 border-red-500 text-red-900'
                    : 'bg-slate-50 border-slate-200 text-slate-400'
                : selected === idx
                  ? 'bg-brand-50 border-brand-500 text-brand-900 shadow-md ring-1 ring-brand-500'
                  : 'bg-white border-slate-200 hover:bg-slate-50 hover:border-slate-300'
            }`}
          >
            <div className="flex items-center gap-3">
              <div className={`w-6 h-6 rounded-full border flex items-center justify-center text-xs font-bold ${
                submitted && idx === question.correctAnswer ? 'bg-green-500 text-white border-green-500' :
                submitted && idx === selected ? 'bg-red-500 text-white border-red-500' :
                selected === idx ? 'bg-brand-500 text-white border-brand-500' :
                'bg-white border-slate-300 text-slate-500'
              }`}>
                {String.fromCharCode(65 + idx)}
              </div>
              <span>{option}</span>
            </div>
            
            {submitted && idx === question.correctAnswer && (
              <CheckCircle size={20} className="absolute right-4 top-1/2 -translate-y-1/2 text-green-600" />
            )}
            {submitted && idx === selected && idx !== question.correctAnswer && (
              <XCircle size={20} className="absolute right-4 top-1/2 -translate-y-1/2 text-red-500" />
            )}
          </button>
        ))}
      </div>

      {!submitted ? (
        <div className="mt-6 flex justify-end">
          <button
            onClick={handleSubmit}
            disabled={selected === null}
            className="px-6 py-2 bg-slate-900 text-white font-bold rounded-lg hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {t.check}
          </button>
        </div>
      ) : (
        <div className={`mt-6 p-4 rounded-lg animate-in fade-in slide-in-from-top-2 ${isCorrect ? 'bg-green-100/50' : 'bg-red-100/50'}`}>
          <div className="flex items-start gap-3">
            <div className="flex-1">
              <h4 className={`font-bold mb-1 ${isCorrect ? 'text-green-800' : 'text-red-800'}`}>
                {isCorrect ? t.correct : t.incorrect}
              </h4>
              <p className="text-slate-700 text-sm">
                <span className="font-bold">{t.why}</span> {question.explanation}
              </p>
            </div>
            {!isCorrect && (
               <button 
                 onClick={handleRetry}
                 className="text-xs font-bold underline text-slate-500 hover:text-slate-800"
               >
                 {t.retry}
               </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default QuizSection;