import React from 'react';
import { Chapter, Language } from './types';
import MathBlock from './components/MathBlock';
import InteractiveAttention from './components/InteractiveAttention';
import ArchitectureDiagram from './components/ArchitectureDiagram';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Helper for code blocks
const Code = ({ code }: { code: string }) => (
  <div className="my-4">
    <div className="bg-slate-800 text-slate-300 px-4 py-2 text-xs rounded-t-lg font-mono border-b border-slate-700">
      PyTorch Implementation / Pseudo-code
    </div>
    <pre className="bg-slate-900 text-blue-100 p-4 rounded-b-lg overflow-x-auto text-sm font-mono leading-relaxed shadow-inner">
      <code>{code}</code>
    </pre>
  </div>
);

const perfData = [
  { name: 'ByteNet', bleu: 23.7, speed: 10 },
  { name: 'Deep-Att', bleu: 24.6, speed: 20 },
  { name: 'GNMT', bleu: 24.6, speed: 50 },
  { name: 'Transformer (Base)', bleu: 27.3, speed: 90 },
  { name: 'Transformer (Big)', bleu: 28.4, speed: 80 },
];

export const uiTranslations = {
  en: {
    home: "Home",
    curriculum: "Curriculum",
    start: "Start Learning",
    paper: "Read Paper",
    prev: "Previous",
    next: "Next",
    title: "Transformer Explained",
    subtitle: "Transformer Decoded",
    desc: "A comprehensive, interactive deep dive into the architecture that revolutionized Natural Language Processing. Based on the paper \"Attention Is All You Need\".",
    whyTitle: "Why this matters",
    whyDesc: "The Transformer abandoned the recurrence of RNNs and introduced a purely attention-based architecture. This shift enabled massive parallelization, paving the way for models like BERT, GPT-4, and Gemini.",
    stats: { published: "Published", citations: "Citations", concept: "Key Concept" }
  },
  zh: {
    home: "首页",
    curriculum: "课程大纲",
    start: "开始学习",
    paper: "阅读论文",
    prev: "上一章",
    next: "下一章",
    title: "Transformer 详解",
    subtitle: "Transformer 解密",
    desc: "深入浅出地解析改变自然语言处理格局的 Transformer 架构。基于论文《Attention Is All You Need》。",
    whyTitle: "核心意义",
    whyDesc: "Transformer 摒弃了 RNN 的循环结构，引入了纯粹的注意力机制。这一转变为 BERT、GPT-4 和 Gemini 等模型的大规模并行化铺平了道路。",
    stats: { published: "发布于", citations: "引用次数", concept: "核心概念" }
  }
};

const chaptersEn: Chapter[] = [
  {
    id: 'ch1',
    title: '1. Background & Motivation',
    description: 'Why do we need the Transformer?',
    sections: [
      {
        id: '1-1',
        title: 'Pain Points of Recurrent Models',
        content: (
          <div>
            <p className="mb-4">
              Before 2017, sequence modeling (like translation) was dominated by RNNs and LSTMs.
              These models process data <strong>sequentially</strong>.
            </p>
            <ul className="list-disc ml-6 space-y-2 mb-4">
              <li><strong>Sequential Computation:</strong> To compute hidden state <MathBlock formula="h_t" />, you need <MathBlock formula="h_{t-1}" />. This precludes parallelization.</li>
              <li><strong>Long-Term Dependencies:</strong> Information from the beginning of a long sentence often fades before reaching the end.</li>
            </ul>
            <div className="bg-red-50 border-l-4 border-red-500 p-4">
              <strong>The Goal:</strong> Create a model that is highly parallelizable and can relate any two positions in a sequence instantly.
            </div>
          </div>
        )
      },
      {
        id: '1-2',
        title: 'Attention is All You Need',
        content: (
          <div>
            <p className="mb-4">
              The paper proposes that we don't need recurrence (RNNs) or convolution (CNNs). Instead, we can rely entirely on an <strong>Attention Mechanism</strong> to draw global dependencies between input and output.
            </p>
          </div>
        )
      }
    ]
  },
  {
    id: 'ch2',
    title: '2. Architecture Overview',
    description: 'The High-Level Encoder-Decoder Structure',
    sections: [
      {
        id: '2-1',
        title: 'The Big Picture',
        content: (
          <div>
            <p className="mb-4">
              The Transformer follows an Encoder-Decoder architecture.
            </p>
            <ArchitectureDiagram lang="en" />
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded shadow-sm border">
                <h4 className="font-bold text-brand-600 mb-2">Encoder (Left)</h4>
                <p className="text-sm">Takes the input sequence (e.g., English sentence) and maps it to a continuous representation holding the meaning.</p>
              </div>
              <div className="bg-white p-4 rounded shadow-sm border">
                <h4 className="font-bold text-pink-600 mb-2">Decoder (Right)</h4>
                <p className="text-sm">Takes the Encoder's output and generates the target sequence (e.g., French translation) one element at a time.</p>
              </div>
            </div>
          </div>
        )
      }
    ]
  },
  {
    id: 'ch3',
    title: '3. Core Components (Deep Dive)',
    description: 'Understanding the mechanics',
    sections: [
      {
        id: '3-1',
        title: 'Self-Attention',
        difficulty: 'advanced',
        content: (
          <div>
            <p className="mb-4">
              The heart of the Transformer. It allows the model to look at other words in the input sentence to better understand the current word.
            </p>
            <div className="bg-blue-50 p-4 rounded-lg mb-4 border border-blue-100">
                <h4 className="font-bold text-blue-800 mb-2">Example:</h4>
                <p className="italic text-slate-700">"The animal didn't cross the street because <strong>it</strong> was too tired."</p>
                <p className="mt-2 text-sm">
                    When the model processes the word "it", Self-Attention allows it to associate "it" strongly with "animal". 
                    Without this, the model wouldn't know if "it" referred to the street or the animal.
                </p>
            </div>
            <p className="mb-4">
              For every input token, we create three vectors: 
              <strong>Query ($Q$)</strong>, <strong>Key ($K$)</strong>, and <strong>Value ($V$)</strong>.
            </p>
            <MathBlock formula="\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V" block />
            
            <Code code={`# PyTorch-like Pseudo-code
def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)
    
    # 1. Calculate Scores (How much to focus)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 2. Scale (Stability)
    scores = scores / math.sqrt(d_k)
    
    # 3. Probability (0 to 1)
    weights = F.softmax(scores, dim=-1)
    
    # 4. Weighted Sum
    output = torch.matmul(weights, value)
    return output`} />

            <InteractiveAttention lang="en" />
          </div>
        )
      },
      {
        id: '3-2',
        title: 'Multi-Head Attention',
        content: (
          <div>
            <p className="mb-4">
              Instead of performing a single attention function, we do it $h$ times in parallel with different linear projections. This allows the model to attend to information from different representation subspaces.
            </p>
            <div className="p-4 bg-brand-50 rounded-lg text-sm text-brand-900 mb-4">
              <strong>Analogy:</strong> Imagine reading a book with 8 different colored highlighters. 
              The "Yellow" head focuses on dates, the "Blue" head focuses on names, and the "Green" head focuses on actions.
              Combining them gives you a complete understanding.
            </div>
            <Code code={`class MultiHeadAttention(nn.Module):
    def forward(self, x):
        # Split input into 'h' heads
        # ... (splitting logic)
        
        # Apply attention to each head independently
        head_outputs = [attention(q, k, v) for q,k,v in heads]
        
        # Concatenate results and pass through linear layer
        concat = torch.cat(head_outputs, dim=-1)
        return self.final_linear(concat)`} />
          </div>
        )
      },
      {
        id: '3-3',
        title: 'Positional Encoding',
        difficulty: 'intermediate',
        content: (
          <div>
            <p className="mb-4">
              Since the Transformer has no recurrence, it has no notion of "order". We must inject position info directly into the embeddings.
            </p>
            <div className="bg-yellow-50 p-4 rounded-lg mb-4 border border-yellow-100">
                <h4 className="font-bold text-yellow-800 mb-2">Analogy:</h4>
                <p className="text-sm">
                    Imagine a library where books are thrown in a pile (Bag of Words). You don't know the story order.
                    Positional Encoding is like stamping a page number on each word so the model knows where it belongs, 
                    even if processed simultaneously.
                </p>
            </div>
            <MathBlock formula="PE_{(pos, 2i)} = \\sin(pos / 10000^{2i/d_{\\text{model}}})" block />
            <MathBlock formula="PE_{(pos, 2i+1)} = \\cos(pos / 10000^{2i/d_{\\text{model}}})" block />
            <Code code={`# Adding position info to embeddings
position = torch.arange(max_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) * ...)

pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

# Add directly to input embeddings
x = embeddings(input) + pe`} />
          </div>
        )
      }
    ]
  },
  {
    id: 'ch4',
    title: '4. Training & Details',
    description: 'How to make it learn',
    sections: [
      {
        id: '4-1',
        title: 'Masking',
        content: (
          <div>
            <h3 className="font-bold text-lg mb-2">Technical Explanation</h3>
            <p className="mb-4">
              <strong>Padding Mask:</strong> Ignores padding tokens (usually index 0) so they don't affect gradients.
              <br/>
              <strong>Look-Ahead Mask:</strong> Used in the decoder. When predicting token at $t$, it masks tokens at $t+1$ and beyond by setting their attention scores to $-\infty$.
            </p>
            
            <div className="bg-slate-100 p-4 rounded-lg border-l-4 border-slate-400">
                <h4 className="font-bold mb-1">Layman's Understanding</h4>
                <p className="text-sm">
                    Imagine taking a test. You want to learn to predict the next word.
                    If you can see the future words (the answers) while guessing the current one, you aren't learning.
                    <strong>Look-Ahead Masking</strong> is like covering the rest of the sentence with a piece of paper so you can only see what you've written so far.
                </p>
            </div>
          </div>
        )
      },
      {
        id: '4-2',
        title: 'Optimizer & Regularization',
        content: (
          <div>
            <h3 className="font-bold text-lg mb-2">Technical Explanation</h3>
            <p className="mb-4">
              Uses the <strong>Adam</strong> optimizer with custom $\beta$ parameters.
              Crucially, it uses a <strong>Learning Rate Schedule</strong> with a "warmup" phase.
              Regularization includes <strong>Residual Connections</strong> (Skip connections) and <strong>Layer Normalization</strong>.
            </p>
            
            <div className="bg-slate-100 p-4 rounded-lg border-l-4 border-slate-400">
                <h4 className="font-bold mb-1">Layman's Understanding</h4>
                <p className="text-sm mb-2">
                    <strong>Warmup:</strong> It's like a sprinter starting a race. You don't go full speed immediately; you accelerate smoothly to avoid stumbling (diverging gradients), then settle into a pace.
                </p>
                <p className="text-sm">
                   <strong>Residual Connections:</strong> Like having a "direct highway" for information to flow through the network, preventing it from getting lost in the complex "traffic" of the layers.
                </p>
            </div>
          </div>
        )
      }
    ]
  },
  {
    id: 'ch5',
    title: '5. Experimental Results',
    description: 'Does it actually work?',
    sections: [
      {
        id: '5-1',
        title: 'Performance & Efficiency',
        content: (
          <div>
            <p className="mb-4">
                The Transformer achieved state-of-the-art results on English-to-German and English-to-French translation tasks (WMT 2014), while requiring significantly less training time than RNN/CNN based predecessors.
            </p>
            <div className="h-64 w-full mb-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={perfData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[20, 30]} />
                  <YAxis dataKey="name" type="category" width={100} style={{fontSize: '12px'}} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="bleu" name="BLEU Score" fill="#0ea5e9" />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-center text-xs text-slate-500 mt-2">Comparison of BLEU scores (Higher is better)</p>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
                <div className="p-4 border rounded bg-white">
                    <h4 className="font-bold text-green-600">Training Efficiency</h4>
                    <p className="text-sm mt-1">
                        Because of parallelization, the Transformer (Big) trained in just <strong>3.5 days</strong> on 8 GPUs, whereas previous best models took weeks.
                    </p>
                </div>
                <div className="p-4 border rounded bg-white">
                    <h4 className="font-bold text-purple-600">Generalization</h4>
                    <p className="text-sm mt-1">
                        The paper demonstrated the model generalizes well to other tasks like Constituency Parsing with minimal tuning.
                    </p>
                </div>
            </div>
          </div>
        )
      }
    ]
  },
  {
    id: 'ch6',
    title: '6. Resources',
    description: 'Further exploration',
    sections: [
      {
        id: '6-1',
        title: 'Official Implementations',
        content: (
          <div>
            <p className="mb-4">
                The original code was released in the Tensor2Tensor library. Below are the links to the original implementations and the paper.
            </p>
            <ul className="space-y-4">
                <li>
                    <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer" className="flex items-center gap-2 p-4 border rounded hover:bg-slate-50 transition group">
                        <span className="text-2xl">📄</span>
                        <div>
                            <div className="font-bold text-brand-600 group-hover:underline">Original Paper (ArXiv)</div>
                            <div className="text-sm text-slate-500">Attention Is All You Need (Vaswani et al., 2017)</div>
                        </div>
                    </a>
                </li>
                <li>
                    <a href="https://github.com/tensorflow/tensor2tensor" target="_blank" rel="noreferrer" className="flex items-center gap-2 p-4 border rounded hover:bg-slate-50 transition group">
                        <span className="text-2xl">💻</span>
                        <div>
                            <div className="font-bold text-slate-800 group-hover:underline">Tensor2Tensor (Original Code)</div>
                            <div className="text-sm text-slate-500">The official TensorFlow implementation used in the paper.</div>
                        </div>
                    </a>
                </li>
                 <li>
                    <a href="https://github.com/pytorch/fairseq" target="_blank" rel="noreferrer" className="flex items-center gap-2 p-4 border rounded hover:bg-slate-50 transition group">
                        <span className="text-2xl">🔥</span>
                        <div>
                            <div className="font-bold text-slate-800 group-hover:underline">PyTorch FairSeq</div>
                            <div className="text-sm text-slate-500">Facebook AI Research Sequence-to-Sequence Toolkit.</div>
                        </div>
                    </a>
                </li>
            </ul>
          </div>
        )
      }
    ]
  }
];

const chaptersZh: Chapter[] = [
  {
    id: 'ch1',
    title: '1. 背景与动机',
    description: '为什么我们需要 Transformer？',
    sections: [
      {
        id: '1-1',
        title: '传统模型的痛点',
        content: (
          <div>
            <p className="mb-4">
              在2017年之前，序列建模（如机器翻译）主要由 RNN 和 LSTM 主导。
              这些模型以<strong>串行方式</strong>处理数据。
            </p>
            <ul className="list-disc ml-6 space-y-2 mb-4">
              <li><strong>串行计算：</strong> 计算隐藏状态 <MathBlock formula="h_t" /> 必须依赖 <MathBlock formula="h_{t-1}" />。这使得无法进行并行计算。</li>
              <li><strong>长距离依赖：</strong> 在处理长句子时，开头的信息往往在到达句子末尾时已经丢失。</li>
            </ul>
            <div className="bg-red-50 border-l-4 border-red-500 p-4">
              <strong>目标：</strong> 创建一个高度并行化的模型，并能瞬间建立序列中任意两个位置的联系。
            </div>
          </div>
        )
      },
      {
        id: '1-2',
        title: 'Attention Is All You Need',
        content: (
          <div>
            <p className="mb-4">
              论文提出我们不再需要循环（RNN）或卷积（CNN）。相反，我们可以完全依赖<strong>注意力机制 (Attention Mechanism)</strong> 来捕捉输入与输出之间的全局依赖关系。
            </p>
          </div>
        )
      }
    ]
  },
  {
    id: 'ch2',
    title: '2. 整体架构概览',
    description: '宏观视角的编码器-解码器结构',
    sections: [
      {
        id: '2-1',
        title: '全景图',
        content: (
          <div>
            <p className="mb-4">
              Transformer 遵循编码器-解码器 (Encoder-Decoder) 架构。
            </p>
            <ArchitectureDiagram lang="zh" />
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded shadow-sm border">
                <h4 className="font-bold text-brand-600 mb-2">Encoder (编码器 - 左侧)</h4>
                <p className="text-sm">接收输入序列（例如：英文句子）并将其映射为包含语义信息的连续表示。</p>
              </div>
              <div className="bg-white p-4 rounded shadow-sm border">
                <h4 className="font-bold text-pink-600 mb-2">Decoder (解码器 - 右侧)</h4>
                <p className="text-sm">接收编码器的输出，并逐个元素生成目标序列（例如：法文翻译）。</p>
              </div>
            </div>
          </div>
        )
      }
    ]
  },
  {
    id: 'ch3',
    title: '3. 核心组件详解',
    description: '深入剖析内部机制',
    sections: [
      {
        id: '3-1',
        title: '自注意力机制 (Self-Attention)',
        difficulty: 'advanced',
        content: (
          <div>
            <p className="mb-4">
              这是 Transformer 的核心。它允许模型在处理当前词时，关注输入句子中的其他词，从而更好地理解上下文。
            </p>
            <div className="bg-blue-50 p-4 rounded-lg mb-4 border border-blue-100">
                <h4 className="font-bold text-blue-800 mb-2">举个栗子：</h4>
                <p className="italic text-slate-700">"The animal didn't cross the street because <strong>it</strong> was too tired."</p>
                <p className="mt-2 text-sm">
                    当模型处理 "it"（它）这个词时，自注意力机制会将它与 "animal"（动物）强烈关联起来。
                    如果没有这个机制，机器可能不知道 "it" 指的是街道还是动物。
                </p>
            </div>

            <p className="mb-4">
              对于每个输入 Token，我们创建三个向量：
              <strong>Query ($Q$, 查询)</strong>, <strong>Key ($K$, 键)</strong>, 和 <strong>Value ($V$, 值)</strong>。
            </p>
            <MathBlock formula="\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V" block />
            
            <Code code={`# PyTorch 风格伪代码
def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)
    
    # 1. 计算分数 (关注度) - 两个向量越相似，点积越大
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 2. 缩放 (保持梯度稳定)
    scores = scores / math.sqrt(d_k)
    
    # 3. 概率归一化 (Softmax 保证和为 1)
    weights = F.softmax(scores, dim=-1)
    
    # 4. 加权求和 (提取信息)
    output = torch.matmul(weights, value)
    return output`} />

            <InteractiveAttention lang="zh" />
          </div>
        )
      },
      {
        id: '3-2',
        title: '多头注意力 (Multi-Head Attention)',
        content: (
          <div>
            <p className="mb-4">
              模型不是只执行一次注意力函数，而是并行执行 $h$ 次，每次使用不同的线性投影。这允许模型关注不同表示子空间的信息。
            </p>
            <div className="p-4 bg-brand-50 rounded-lg text-sm text-brand-900 mb-4">
              <strong>通俗理解：</strong> 就像看书时使用 8 种不同颜色的荧光笔。
              黄色笔标记“时间”，蓝色笔标记“人物”，绿色笔标记“动作”。
              最后把所有标记的信息汇总，你就得到了最全面的理解。如果只有一种颜色，信息可能会混杂。
            </div>
             <Code code={`class MultiHeadAttention(nn.Module):
    def forward(self, x):
        # 将输入分割成 h 个头 (Heads)
        # ... (split logic)
        
        # 每个头独立进行注意力计算
        head_outputs = [attention(q, k, v) for q,k,v in heads]
        
        # 拼接所有头的结果并通过线性层融合
        concat = torch.cat(head_outputs, dim=-1)
        return self.final_linear(concat)`} />
          </div>
        )
      },
      {
        id: '3-3',
        title: '位置编码 (Positional Encoding)',
        difficulty: 'intermediate',
        content: (
          <div>
             <p className="mb-4">
              由于 Transformer 没有循环结构，它本身不知道单词的顺序。我们必须将位置信息注入到 Embedding 中。
            </p>
             <div className="bg-yellow-50 p-4 rounded-lg mb-4 border border-yellow-100">
                <h4 className="font-bold text-yellow-800 mb-2">通俗理解：</h4>
                <p className="text-sm">
                    想象图书馆把一本书拆散成一堆纸（Bag of Words），顺序全乱了。
                    位置编码就像是在每一页纸的页脚打上页码。这样即使你同时处理所有纸张，你也知道哪一页在前，哪一页在后。
                </p>
            </div>
            <p className="mb-4">
              我们通过使用不同频率的正弦和余弦函数，将位置向量添加到输入嵌入中。
            </p>
            <MathBlock formula="PE_{(pos, 2i)} = \\sin(pos / 10000^{2i/d_{\\text{model}}})" block />
            <MathBlock formula="PE_{(pos, 2i+1)} = \\cos(pos / 10000^{2i/d_{\\text{model}}})" block />
            <Code code={`# 生成位置编码并加到 Input Embedding 上
position = torch.arange(max_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) * ...)

pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

# 直接相加，不改变维度
x = embeddings(input) + pe`} />
          </div>
        )
      }
    ]
  },
  {
    id: 'ch4',
    title: '4. 训练与细节',
    description: '如何让模型学习',
    sections: [
      {
        id: '4-1',
        title: '掩码 (Masking)',
        content: (
          <div>
            <h3 className="font-bold text-lg mb-2">技术解释</h3>
            <p className="mb-4">
              <strong>填充掩码 (Padding Mask)：</strong> 忽略输入序列中为了对齐长度而填充的 0 (Padding Tokens)。
              <br/>
              <strong>前瞻掩码 (Look-Ahead Mask)：</strong> 用于解码器。在预测第 $t$ 个词时，将 $t$ 之后的位置的注意力分数设为 $-\\infty$（负无穷）。
            </p>
            
            <div className="bg-slate-100 p-4 rounded-lg border-l-4 border-slate-400">
                <h4 className="font-bold mb-1">通俗理解：</h4>
                <p className="text-sm">
                    这就好比在做英语填空题。
                    如果你在填第 3 个空的时候，偷看了第 4 个空的答案，那你就没有真正学会预测。
                    <strong>前瞻掩码</strong>就像是用一张纸把后面的答案挡住，强迫模型只能根据已知的上文来推断下文。
                </p>
            </div>
          </div>
        )
      },
      {
        id: '4-2',
        title: '优化器与正则化',
        content: (
          <div>
             <h3 className="font-bold text-lg mb-2">技术解释</h3>
            <p className="mb-4">
              使用 <strong>Adam</strong> 优化器。关键在于使用了 <strong>Warmup（热身）</strong> 策略：学习率在训练初期线性增加，随后按平方根倒数衰减。
              此外广泛使用了残差连接 (Residual Connections) 和层归一化 (Layer Normalization)。
            </p>
            
             <div className="bg-slate-100 p-4 rounded-lg border-l-4 border-slate-400">
                <h4 className="font-bold mb-1">通俗理解：</h4>
                <p className="text-sm mb-2">
                    <strong>Warmup (热身)：</strong> 就像开车上高速。你不能一启动就挂五档（大学习率），那样容易熄火（梯度发散）。你需要慢慢加速（线性增加），等车跑顺了再稳定速度巡航。
                </p>
                <p className="text-sm">
                   <strong>残差连接：</strong> 就像给信息传达开了一条“快速通道”。即使网络很深，信息也可以通过这条通道直接传到深层，防止在层层传递中丢失（梯度消失）。
                </p>
            </div>
          </div>
        )
      }
    ]
  },
  {
    id: 'ch5',
    title: '5. 实验结果与影响',
    description: '它真的有效吗？',
    sections: [
      {
        id: '5-1',
        title: '多维度性能分析',
        content: (
          <div>
             <p className="mb-4">
                Transformer 在 WMT 2014 英德和英法翻译任务上均达到了当时的 SOTA (State-of-the-art) 水平，且训练成本大幅降低。
            </p>
            <div className="h-64 w-full mb-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={perfData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[20, 30]} />
                  <YAxis dataKey="name" type="category" width={100} style={{fontSize: '12px'}} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="bleu" name="BLEU 分数" fill="#0ea5e9" />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-center text-xs text-slate-500 mt-2">BLEU 分数对比（越高越好）</p>
            </div>
            
             <div className="grid grid-cols-2 gap-4">
                <div className="p-4 border rounded bg-white">
                    <h4 className="font-bold text-green-600">训练效率 (Efficiency)</h4>
                    <p className="text-sm mt-1">
                        得益于并行计算，Transformer (Big) 在 8 张 GPU 上仅训练了 <strong>3.5 天</strong>。相比之下，之前的 LSTM 模型往往需要训练数周。
                    </p>
                </div>
                <div className="p-4 border rounded bg-white">
                    <h4 className="font-bold text-purple-600">泛化能力 (Generalization)</h4>
                    <p className="text-sm mt-1">
                        论文不仅测试了翻译，还证明了该模型可以很好地迁移到其他任务，如成分句法分析 (Constituency Parsing)，且几乎不需要调整超参数。
                    </p>
                </div>
            </div>
          </div>
        )
      }
    ]
  },
  {
    id: 'ch6',
    title: '6. 资源与代码',
    description: '深入研究',
    sections: [
      {
        id: '6-1',
        title: '官方实现与源码',
        content: (
          <div>
            <p className="mb-4">
                Transformer 的原始代码发布在 Google 的 Tensor2Tensor 库中。以下是重要资源的链接：
            </p>
            <ul className="space-y-4">
                <li>
                    <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer" className="flex items-center gap-2 p-4 border rounded hover:bg-slate-50 transition group">
                        <span className="text-2xl">📄</span>
                        <div>
                            <div className="font-bold text-brand-600 group-hover:underline">阅读原始论文 (ArXiv)</div>
                            <div className="text-sm text-slate-500">Attention Is All You Need (Vaswani et al., 2017)</div>
                        </div>
                    </a>
                </li>
                <li>
                    <a href="https://github.com/tensorflow/tensor2tensor" target="_blank" rel="noreferrer" className="flex items-center gap-2 p-4 border rounded hover:bg-slate-50 transition group">
                        <span className="text-2xl">💻</span>
                        <div>
                            <div className="font-bold text-slate-800 group-hover:underline">Tensor2Tensor (原始代码)</div>
                            <div className="text-sm text-slate-500">论文使用的官方 TensorFlow 实现。</div>
                        </div>
                    </a>
                </li>
                 <li>
                    <a href="https://github.com/pytorch/fairseq" target="_blank" rel="noreferrer" className="flex items-center gap-2 p-4 border rounded hover:bg-slate-50 transition group">
                        <span className="text-2xl">🔥</span>
                        <div>
                            <div className="font-bold text-slate-800 group-hover:underline">PyTorch FairSeq</div>
                            <div className="text-sm text-slate-500">Facebook AI Research 推出的序列建模工具包，包含高质量的 Transformer 实现。</div>
                        </div>
                    </a>
                </li>
            </ul>
          </div>
        )
      }
    ]
  }
];

export const getChapters = (lang: Language) => lang === 'zh' ? chaptersZh : chaptersEn;