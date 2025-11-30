import React, { useState, useMemo } from 'react';
import MathBlock from './MathBlock';
import { Language } from '../types';

interface Props {
  lang: Language;
}

const InteractiveAttention: React.FC<Props> = ({ lang }) => {
  // Simplified dimensions for visualization
  const [dim, setDim] = useState(3);
  
  const t = {
    en: {
      title: "Interactive: Scaled Dot-Product Attention",
      embedDim: "Embedding Dimension",
      drag: "Drag to change vector size and see how calculations update.",
      q: "Query (Q)",
      k: "Keys (K)",
      v: "Values (V)",
      scores: "Scores",
      weights: "Weights",
      output: "Output",
      formula: "Formula:",
      legend: {
        q: "Query (Blue): Current token representation.",
        k: "Keys (Green): Representations of other tokens to match against.",
        w: "Weights (Red): How much \"attention\" to pay to each Value. Note how they sum to 1.0.",
        o: "Output: The weighted sum of Values."
      }
    },
    zh: {
      title: "交互演示：缩放点积注意力",
      embedDim: "嵌入维度",
      drag: "拖动滑块改变向量维度，观察计算结果如何变化。",
      q: "查询向量 (Q)",
      k: "键向量 (K)",
      v: "值向量 (V)",
      scores: "分数",
      weights: "权重",
      output: "输出",
      formula: "公式：",
      legend: {
        q: "Query (蓝): 当前 Token 的表示。",
        k: "Keys (绿): 用于匹配的其他 Token 的表示。",
        w: "Weights (红): 对每个 Value 的“关注”程度。注意它们的和为 1.0。",
        o: "Output: Values 的加权和。"
      }
    }
  }[lang];

  // Fake data generation based on dimensions
  const generateMatrix = (rows: number, cols: number, seed: number) => {
    return Array.from({ length: rows }, (_, r) => 
      Array.from({ length: cols }, (_, c) => parseFloat(((Math.sin(r * cols + c + seed) + 1) / 2).toFixed(2)))
    );
  };

  const Q = useMemo(() => generateMatrix(1, dim, 1), [dim]);
  const K = useMemo(() => generateMatrix(3, dim, 2), [dim]); // 3 keys
  const V = useMemo(() => generateMatrix(3, dim, 3), [dim]); // 3 values

  // Calculate Attention
  // 1. MatMul Q * K^T
  const scores = K.map(kRow => {
    const dot = kRow.reduce((sum, val, i) => sum + val * Q[0][i], 0);
    return dot;
  });

  // 2. Scale
  const scaledScores = scores.map(s => s / Math.sqrt(dim));

  // 3. Softmax
  const expScores = scaledScores.map(s => Math.exp(s));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const weights = expScores.map(s => s / sumExp);

  // 4. Output = Weights * V
  const output = Array.from({ length: dim }, (_, colIndex) => {
    return weights.reduce((sum, w, rowIndex) => sum + w * V[rowIndex][colIndex], 0);
  });

  const renderMatrix = (matrix: number[][], title: string, color: string) => (
    <div className="flex flex-col items-center mx-2">
      <h4 className="font-bold text-sm mb-2">{title}</h4>
      <div className={`grid gap-1 border-2 border-${color}-200 p-1 rounded bg-white`} 
           style={{ gridTemplateColumns: `repeat(${matrix[0].length}, minmax(0, 1fr))` }}>
        {matrix.flat().map((val, i) => (
          <div key={i} className={`w-8 h-8 flex items-center justify-center text-xs bg-${color}-50 rounded`} title={val.toFixed(2)}>
            {val.toFixed(1)}
          </div>
        ))}
      </div>
    </div>
  );

  const renderVector = (vector: number[], title: string, color: string) => (
    <div className="flex flex-col items-center mx-2">
      <h4 className="font-bold text-sm mb-2">{title}</h4>
      <div className={`grid gap-1 border-2 border-${color}-200 p-1 rounded bg-white`}>
        {vector.map((val, i) => (
          <div key={i} className={`w-8 h-8 flex items-center justify-center text-xs bg-${color}-50 rounded`} title={val.toFixed(4)}>
            {val.toFixed(2)}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="bg-slate-100 p-6 rounded-xl border border-slate-200 my-6 shadow-sm">
      <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
        <span>{t.title}</span>
      </h3>
      
      <div className="mb-6">
        <label className="block text-sm font-medium text-slate-700 mb-1">{t.embedDim} ($d_k$): {dim}</label>
        <input 
          type="range" 
          min="2" 
          max="6" 
          value={dim} 
          onChange={(e) => setDim(parseInt(e.target.value))} 
          className="w-full h-2 bg-slate-300 rounded-lg appearance-none cursor-pointer"
        />
        <p className="text-xs text-slate-500 mt-1">{t.drag}</p>
      </div>

      <div className="flex flex-wrap justify-center items-start gap-4 mb-8">
        {renderMatrix(Q, t.q, "blue")}
        <div className="self-center font-mono text-xl">×</div>
        {renderMatrix(K, t.k, "green")}
        <div className="self-center font-mono text-xl">➜</div>
        {renderVector(scores, t.scores, "yellow")}
        <div className="self-center font-mono text-xl">softmax</div>
        {renderVector(weights, t.weights, "red")}
      </div>

      <div className="flex flex-wrap justify-center items-start gap-4">
        {renderVector(weights, t.weights, "red")}
        <div className="self-center font-mono text-xl">×</div>
        {renderMatrix(V, t.v, "purple")}
        <div className="self-center font-mono text-xl">=</div>
        {renderMatrix([output], t.output, "brand")}
      </div>

      <div className="mt-6 text-sm text-slate-600 bg-white p-4 rounded border border-slate-200">
        <p><strong>{t.formula}</strong> <MathBlock formula="\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V" /></p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>{t.legend.q}</strong></li>
          <li><strong>{t.legend.k}</strong></li>
          <li><strong>{t.legend.w}</strong></li>
          <li><strong>{t.legend.o}</strong></li>
        </ul>
      </div>
    </div>
  );
};

export default InteractiveAttention;