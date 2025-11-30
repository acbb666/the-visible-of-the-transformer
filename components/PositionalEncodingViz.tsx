import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Language } from '../types';
import MathBlock from './MathBlock';

interface Props {
  lang: Language;
}

const PositionalEncodingViz: React.FC<Props> = ({ lang }) => {
  const [dimension, setDimension] = useState(4); // Even dimension index 2i
  
  const d_model = 512;
  const data = [];
  
  // Generate data for positions 0 to 50
  for (let pos = 0; pos <= 50; pos++) {
    // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    const wavelength = Math.pow(10000, (dimension / d_model));
    const val = Math.sin(pos / wavelength);
    data.push({ pos, val });
  }

  const t = {
    en: {
      title: "Visualizing Frequency",
      desc: "Adjust the Dimension Index (2i) to see how the frequency of the Sine wave changes. Lower dimensions have high frequency; higher dimensions have low frequency.",
      dimIndex: "Dimension Index (2i)",
      pos: "Position in Sequence",
      val: "Encoding Value"
    },
    zh: {
      title: "可视化频率变化",
      desc: "调整维度索引 (2i) 来观察正弦波的频率如何变化。低维度具有高频率（变化快），高维度具有低频率（变化慢）。",
      dimIndex: "维度索引 (2i)",
      pos: "序列位置",
      val: "编码值"
    }
  }[lang];

  return (
    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm my-6">
      <h3 className="font-bold text-slate-800 mb-4">{t.title}</h3>
      
      <div className="mb-6">
        <div className="flex justify-between text-sm mb-2">
           <label>{t.dimIndex}: <span className="font-mono font-bold text-brand-600">{dimension}</span></label>
           <span className="text-slate-400">0 - {d_model}</span>
        </div>
        <input 
          type="range" 
          min="0" 
          max="512" 
          step="2"
          value={dimension} 
          onChange={(e) => setDimension(parseInt(e.target.value))} 
          className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
        />
        <div className="flex justify-between text-xs text-slate-400 mt-1">
            <span>High Freq (Fast)</span>
            <span>Low Freq (Slow)</span>
        </div>
      </div>

      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
            <XAxis dataKey="pos" label={{ value: t.pos, position: 'insideBottomRight', offset: -5, fontSize: 10 }} fontSize={10} />
            <YAxis domain={[-1.2, 1.2]} fontSize={10} />
            <Tooltip />
            <ReferenceLine y={0} stroke="#94a3b8" />
            <Line type="monotone" dataKey="val" stroke="#0284c7" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 p-3 bg-slate-50 rounded text-xs text-slate-600 border border-slate-100">
        <MathBlock formula={`PE(pos, ${dimension}) = \\sin(pos / 10000^{${dimension}/${d_model}})`} />
      </div>
      
      <p className="text-sm text-slate-600 mt-2">{t.desc}</p>
    </div>
  );
};

export default PositionalEncodingViz;