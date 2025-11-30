import React from 'react';
import { Chapter, Language, QuizQuestion } from './types';
import MathBlock from './components/MathBlock';
import InteractiveAttention from './components/InteractiveAttention';
import ArchitectureDiagram from './components/ArchitectureDiagram';
import TokenizationDemo from './components/TokenizationDemo';
import PositionalEncodingViz from './components/PositionalEncodingViz';
import QuizSection from './components/QuizSection';
import CodeBlock from './components/CodeBlock';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

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

// --- DATA DEFINITIONS ---

const chaptersEn: Chapter[] = [
  {
    id: 'ch0',
    title: '0. Inputs & Embeddings',
    description: 'Before the architecture: How machines read text',
    sections: [
      {
        id: '0-1',
        title: 'From Text to Numbers',
        difficulty: 'basic',
        content: (
          <div>
            <p className="mb-4">
              Computers cannot understand raw text like "Hello". They can only process numbers. 
              Therefore, the first step in any NLP model is <strong>Tokenization</strong>.
            </p>
            <p className="mb-4">
              We break down a sentence into smaller chunks called "Tokens". These can be words, characters, or sub-words.
            </p>
            <TokenizationDemo lang="en" />
            <p className="mb-4">
              Once tokenized, each token is assigned a unique integer ID from a vocabulary.
            </p>
          </div>
        )
      },
      {
        id: '0-2',
        title: 'Input Embeddings',
        content: (
          <div>
            <p className="mb-4">
              Integer IDs are not enough (e.g., ID 100 is not "twice" ID 50). We convert these integers into dense vectors of size <MathBlock formula="d_{model}" /> (usually 512).
            </p>
            <p className="mb-4">
              These embeddings are learned parameters. During training, the model learns that words with similar meanings (like "King" and "Queen") should have similar vector representations in this 512-dimensional space.
            </p>
            <MathBlock formula="X_{\text{embedding}} = \text{EmbeddingLookup}(x_{\text{input}}) \times \sqrt{d_{model}}" block />
            <p className="text-sm text-slate-500">Note: In the Transformer paper, embeddings are multiplied by <MathBlock formula="\sqrt{d_{model}}" /> to stabilize variance before adding positional encoding.</p>
            
            <QuizSection 
              lang="en"
              question={{
                id: 'q0',
                question: 'Why do we use Embeddings instead of One-Hot Encoding?',
                options: [
                  'Embeddings are easier to calculate.',
                  'Embeddings capture semantic relationships and are less sparse.',
                  'One-Hot vectors are too small.',
                  'Embeddings remove the need for tokenization.'
                ],
                correctAnswer: 1,
                explanation: 'One-hot vectors are extremely sparse and high-dimensional, and they treat every word as equidistant. Dense embeddings capture semantic similarity (e.g., dog is close to cat).'
              }} 
            />
          </div>
        )
      }
    ]
  },
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
            
            <h4 className="font-bold mt-6 mb-2">Complexity Comparison</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm text-left">
                <thead className="bg-slate-100 font-bold">
                  <tr>
                    <th className="p-2">Layer Type</th>
                    <th className="p-2">Complexity per Layer</th>
                    <th className="p-2">Sequential Ops</th>
                    <th className="p-2">Max Path Length</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b">
                    <td className="p-2">RNN</td>
                    <td className="p-2"><MathBlock formula="O(n \cdot d^2)" /></td>
                    <td className="p-2"><MathBlock formula="O(n)" /></td>
                    <td className="p-2"><MathBlock formula="O(n)" /></td>
                  </tr>
                  <tr>
                    <td className="p-2 bg-brand-50 font-bold text-brand-700">Self-Attention</td>
                    <td className="p-2 bg-brand-50"><MathBlock formula="O(n^2 \cdot d)" /></td>
                    <td className="p-2 bg-brand-50"><MathBlock formula="O(1)" /></td>
                    <td className="p-2 bg-brand-50"><MathBlock formula="O(1)" /></td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-xs text-slate-500 mt-2">
              <MathBlock formula="n" /> is sequence length, <MathBlock formula="d" /> is representation dimension.
              Self-Attention is faster for shorter sequences where <MathBlock formula="n < d" />.
            </p>
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
            <QuizSection 
              lang="en"
              question={{
                id: 'q1',
                question: 'What is the primary advantage of Transformer over RNNs regarding training?',
                options: [
                  'It has fewer parameters.',
                  'It allows significant parallelization (O(1) sequential ops).',
                  'It uses Convolutional Neural Networks.',
                  'It requires no data preprocessing.'
                ],
                correctAnswer: 1,
                explanation: 'Because the Transformer processes the entire sequence at once using Attention (instead of step-by-step), it can fully utilize modern GPU parallelism.'
              }} 
            />
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
                <p className="text-sm">Takes the input sequence (e.g., English sentence) and maps it to a continuous representation holding the meaning. It consists of a stack of $N=6$ identical layers.</p>
              </div>
              <div className="bg-white p-4 rounded shadow-sm border">
                <h4 className="font-bold text-pink-600 mb-2">Decoder (Right)</h4>
                <p className="text-sm">Takes the Encoder's output and generates the target sequence (e.g., French translation) one element at a time. It also has $N=6$ layers.</p>
              </div>
            </div>
            
            <QuizSection 
              lang="en"
              question={{
                id: 'q2',
                question: 'What information does the Decoder receive?',
                options: [
                  'Only the target sentence.',
                  'Only the source sentence.',
                  'The output of the Encoder AND the target generated so far.',
                  'Random noise.'
                ],
                correctAnswer: 2,
                explanation: 'The Decoder has two sources of info: Self-Attention (looking at what it has generated so far) and Cross-Attention (looking at the Encoder output).'
              }} 
            />
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
            
            <CodeBlock code={`# 1. Scaled Dot-Product Attention
def attention(query, key, value):
    d_k = query.size(-1)
    
    # Matmul Q and K -> Scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Scale by sqrt(d_k)
    scores = scores / math.sqrt(d_k)
    
    # Softmax to get probabilities
    attn_weights = F.softmax(scores, dim=-1)
    
    # Multiply by V
    return torch.matmul(attn_weights, value)`} />

            <InteractiveAttention lang="en" />
            
             <QuizSection 
              lang="en"
              question={{
                id: 'q3-1',
                question: 'In the equation, why do we divide by sqrt(d_k)?',
                options: [
                  'To reduce computation time.',
                  'To prevent the dot products from growing too large, which would push Softmax into regions with small gradients.',
                  'To make the matrix multiplication valid.',
                  'It is an arbitrary constant.'
                ],
                correctAnswer: 1,
                explanation: 'Large dot products result in Softmax outputs close to 0 or 1, where gradients are extremely small (vanishing gradients). Scaling prevents this.'
              }} 
            />
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
            <CodeBlock code={`class MultiHeadAttention(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. Linear projections for Q, K, V
        # Split into 'h' heads
        Q = self.w_q(x).view(batch_size, -1, self.heads, self.d_k)
        K = self.w_k(x).view(batch_size, -1, self.heads, self.d_k)
        V = self.w_v(x).view(batch_size, -1, self.heads, self.d_k)
        
        # 2. Apply attention to each head
        # (Simplified, actually done via matrix ops)
        out = attention(Q, K, V)
        
        # 3. Concatenate and Linear
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(out)`} />
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
            
            <PositionalEncodingViz lang="en" />
          </div>
        )
      },
      {
        id: '3-4',
        title: 'Feed-Forward Networks (FFN)',
        content: (
          <div>
             <p className="mb-4">
              In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.
            </p>
            <p className="mb-4">
              It consists of two linear transformations with a ReLU activation in between.
            </p>
            <MathBlock formula="\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2" block />
            <p className="text-sm text-slate-600 mb-4">
                The dimensionality of input and output is <MathBlock formula="d_{model} = 512" />, and the inner-layer has dimensionality <MathBlock formula="d_{ff} = 2048" />.
            </p>
            <CodeBlock code={`class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        # Expands dimension (512 -> 2048)
        self.w_1 = nn.Linear(d_model, d_ff) 
        # Restores dimension (2048 -> 512)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # ReLU activation in between
        return self.w_2(F.relu(self.w_1(x)))`} />
          </div>
        )
      },
      {
        id: '3-5',
        title: 'Encoder-Decoder Attention',
        difficulty: 'advanced',
        content: (
           <div>
            <p className="mb-4">
              This is the specific layer in the <strong>Decoder</strong> that allows it to look at the <strong>Encoder's</strong> output.
            </p>
            <ul className="list-disc ml-6 space-y-2 mb-4">
                <li><strong>Queries (Q):</strong> Come from the previous decoder layer (what we are currently translating).</li>
                <li><strong>Keys (K) & Values (V):</strong> Come from the Encoder output (the source sentence).</li>
            </ul>
            <div className="bg-purple-50 p-4 rounded-lg mb-4 border border-purple-100">
                <h4 className="font-bold text-purple-800 mb-2">Why?</h4>
                <p className="text-sm">
                   This aligns the translation with the original text. If the Decoder is trying to generate the French word for "student", 
                   this mechanism allows it to focus on the English word "student" encoded by the Encoder.
                </p>
            </div>
           </div>
        )
      },
      {
        id: '3-6',
        title: 'Add & Norm',
        content: (
            <div>
                <p className="mb-4">
                    The output of each sub-layer (Self-Attention, FFN) is calculated as:
                </p>
                <MathBlock formula="\\text{LayerNorm}(x + \\text{Sublayer}(x))" block />
                <p className="mb-4">
                    <strong>Residual Connection (Add):</strong> We add the input $x$ back to the output. This solves the "vanishing gradient" problem in deep networks.
                    <br />
                    <strong>Layer Normalization (Norm):</strong> We normalize the statistics of the hidden vector to stabilize training.
                </p>
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
            <h3 className="font-bold text-lg mb-2 text-slate-800">Technical Explanation</h3>
            <p className="mb-4">
              <strong>Padding Mask:</strong> Used to ignore "pad" tokens (usually index 0) in the input batch so they don't affect gradients.
              <br/>
              <strong>Look-Ahead Mask:</strong> Crucial for the Decoder. When predicting the token at position $t$, the model must not attend to tokens at $t+1$. We set their attention scores to $-\\infty$ before Softmax, resulting in 0 probability.
            </p>
            
            <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-500 mt-4">
                <h4 className="font-bold mb-1 text-green-800">Layman's Understanding</h4>
                <p className="text-sm text-green-900">
                    Imagine taking a fill-in-the-blank test. You want to learn to predict the next word.
                    If you can see the future words (the answers) while guessing the current one, you aren't learning.
                    <strong>Look-Ahead Masking</strong> is like covering the rest of the sentence with a piece of paper so you can only see what you've written so far.
                </p>
            </div>
            
             <QuizSection 
              lang="en"
              question={{
                id: 'q4-1',
                question: 'Why do we need a Look-Ahead Mask in the Decoder but not the Encoder?',
                options: [
                  'Because the Encoder is bidirectional (sees whole sentence), while Decoder generates sequentially.',
                  'The Encoder does not use Self-Attention.',
                  'The Decoder is faster.',
                  'Padding tokens only exist in the Decoder.'
                ],
                correctAnswer: 0,
                explanation: 'The Encoder processes the full source sentence at once. The Decoder is autoregressive, meaning it generates one word at a time and shouldn\'t "cheat" by seeing future words.'
              }} 
            />
          </div>
        )
      },
      {
        id: '4-2',
        title: 'Optimizer & Regularization',
        content: (
          <div>
            <h3 className="font-bold text-lg mb-2 text-slate-800">Optimizer</h3>
            <p className="mb-4">
              The paper uses the <strong>Adam</strong> optimizer with specific $\beta_1=0.9, \beta_2=0.98$.
              Crucially, it uses a <strong>Learning Rate Schedule</strong> with a "warmup" phase.
            </p>
            <MathBlock formula="lrate = d_{\\text{model}}^{-0.5} \\cdot \\min(step\\_num^{-0.5}, step\\_num \\cdot warmup\\_steps^{-1.5})" block />
            
            <h3 className="font-bold text-lg mb-2 mt-6 text-slate-800">Label Smoothing</h3>
            <p className="mb-4">
              Instead of forcing the model to be 100% confident (Target: 1.0 for correct word, 0 for others), we smooth the target distribution.
              <br/>
              If smoothing <MathBlock formula="\epsilon_{ls} = 0.1" />, the correct word gets probability 0.9, and the rest of the probability mass is distributed among other words.
            </p>
            <p className="text-sm text-slate-600 mb-4">
                This hurts perplexity (uncertainty) but improves accuracy and BLEU score by preventing the model from becoming over-confident and overfitting.
            </p>
          </div>
        )
      }
    ]
  },
  {
      id: 'ch5',
      title: '5. Inference & Decoding',
      description: 'Generating text',
      sections: [
          {
              id: '5-1',
              title: 'Auto-Regressive Generation',
              content: (
                  <div>
                      <p className="mb-4">
                          During inference (translation), the model generates words one by one.
                      </p>
                      <ol className="list-decimal ml-6 space-y-2">
                          <li>Pass the source sentence to Encoder.</li>
                          <li>Give Decoder a special <code>&lt;START&gt;</code> token.</li>
                          <li>Decoder outputs probability distribution for the first word.</li>
                          <li>Pick the best word, add it to the input.</li>
                          <li>Repeat until <code>&lt;END&gt;</code> token is produced.</li>
                      </ol>
                  </div>
              )
          },
          {
              id: '5-2',
              title: 'Greedy vs Beam Search',
              content: (
                  <div>
                      <p className="mb-4">
                          <strong>Greedy Search:</strong> Always pick the word with the highest probability at each step. Fast, but can lead to suboptimal sentences (local optimum).
                      </p>
                      <p className="mb-4">
                          <strong>Beam Search:</strong> Keep track of the top $k$ (beam width) most likely sentences at each step. This explores multiple possibilities simultaneously to find a better overall translation.
                      </p>
                       <QuizSection 
                        lang="en"
                        question={{
                          id: 'q5',
                          question: 'What is the main benefit of Beam Search over Greedy Search?',
                          options: [
                            'It is faster.',
                            'It explores multiple potential sentence paths to avoid getting stuck in local optima.',
                            'It uses less memory.',
                            'It doesn\'t require a Decoder.'
                          ],
                          correctAnswer: 1,
                          explanation: 'Greedy search might pick a word that looks good now but leads to a dead end. Beam search keeps options open longer.'
                        }} 
                      />
                  </div>
              )
          }
      ]
  },
  {
      id: 'ch6',
      title: '6. The Transformer Family',
      description: 'Evolution of the architecture',
      sections: [
          {
              id: '6-1',
              title: 'Encoder-Only (BERT)',
              content: (
                  <div>
                      <h4 className="font-bold text-slate-800">BERT (Bidirectional Encoder Representations from Transformers)</h4>
                      <p className="text-sm mb-2">
                          Uses only the <strong>Encoder</strong> stack.
                      </p>
                      <p className="text-sm mb-2">
                          <strong>Goal:</strong> Understanding text. It looks at the whole sentence at once (Bidirectional).
                      </p>
                      <p className="text-sm">
                          <strong>Tasks:</strong> Classification, Sentiment Analysis, Named Entity Recognition.
                      </p>
                  </div>
              )
          },
          {
              id: '6-2',
              title: 'Decoder-Only (GPT)',
              content: (
                  <div>
                      <h4 className="font-bold text-slate-800">GPT (Generative Pre-trained Transformer)</h4>
                      <p className="text-sm mb-2">
                          Uses only the <strong>Decoder</strong> stack (with masked self-attention).
                      </p>
                      <p className="text-sm mb-2">
                          <strong>Goal:</strong> Generating text. It predicts the next word based on previous words.
                      </p>
                      <p className="text-sm">
                          <strong>Tasks:</strong> Text generation, Chatbots, Code completion.
                      </p>
                  </div>
              )
          },
          {
              id: '6-3',
              title: 'Encoder-Decoder (T5 / BART)',
              content: (
                  <div>
                      <h4 className="font-bold text-slate-800">T5 (Text-to-Text Transfer Transformer)</h4>
                      <p className="text-sm mb-2">
                          Uses the full original architecture.
                      </p>
                      <p className="text-sm">
                          <strong>Tasks:</strong> Translation, Summarization (Sequence-to-Sequence tasks).
                      </p>
                  </div>
              )
          }
      ]
  },
  {
    id: 'ch7',
    title: '7. Experimental Results',
    description: 'Does it actually work?',
    sections: [
      {
        id: '7-1',
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
                    <h4 className="font-bold text-green-600">Training Cost</h4>
                    <p className="text-sm mt-1">
                        The Transformer (Base) cost only <strong>$3.3 \cdot 10^{18}$</strong> floating point operations to train.
                        The Big model took just <strong>3.5 days</strong> on 8 P100 GPUs, whereas previous SOTA models took weeks.
                    </p>
                </div>
                <div className="p-4 border rounded bg-white">
                    <h4 className="font-bold text-purple-600">Generalization</h4>
                    <p className="text-sm mt-1">
                        The paper demonstrated the model generalizes well to other tasks. For example, it performed exceptionally well on <strong>English Constituency Parsing</strong> with minimal task-specific tuning, proving it's not just for translation.
                    </p>
                </div>
            </div>
          </div>
        )
      }
    ]
  },
  {
    id: 'ch8',
    title: '8. Resources',
    description: 'Further exploration',
    sections: [
      {
        id: '8-1',
        title: 'Official Implementations',
        content: (
          <div>
            <p className="mb-4">
                The authors released the original code in the Tensor2Tensor library. Below are direct links to the source material.
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
                            <div className="font-bold text-slate-800 group-hover:underline">Tensor2Tensor (GitHub)</div>
                            <div className="text-sm text-slate-500">The official TensorFlow implementation used in the paper.</div>
                        </div>
                    </a>
                </li>
                 <li>
                    <a href="https://github.com/pytorch/fairseq" target="_blank" rel="noreferrer" className="flex items-center gap-2 p-4 border rounded hover:bg-slate-50 transition group">
                        <span className="text-2xl">🔥</span>
                        <div>
                            <div className="font-bold text-slate-800 group-hover:underline">PyTorch FairSeq (GitHub)</div>
                            <div className="text-sm text-slate-500">Facebook AI Research's toolkit, containing high-quality Transformer implementations.</div>
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
    id: 'ch0',
    title: '0. 输入与嵌入',
    description: '在进入架构之前：机器如何阅读文本',
    sections: [
      {
        id: '0-1',
        title: '从文本到数字',
        difficulty: 'basic',
        content: (
          <div>
            <p className="mb-4">
              计算机无法理解像 "Hello" 这样的原始文本，它们只能处理数字。
              因此，任何 NLP 模型的第一步都是 <strong>分词 (Tokenization)</strong>。
            </p>
            <p className="mb-4">
              我们将句子分解成更小的块，称为 "Token"。这些可以是单词、字符或子词。
            </p>
            <TokenizationDemo lang="zh" />
            <p className="mb-4">
              分词后，每个 Token 都会从词汇表中被分配一个唯一的整数 ID。
            </p>
          </div>
        )
      },
      {
        id: '0-2',
        title: '输入嵌入 (Input Embeddings)',
        content: (
          <div>
            <p className="mb-4">
              整数 ID 是不够的（例如，ID 100 并不意味着它是 ID 50 的两倍）。我们将这些整数转换为大小为 <MathBlock formula="d_{model}" />（通常为 512）的密集向量。
            </p>
            <p className="mb-4">
              这些嵌入是可学习的参数。在训练过程中，模型会学习到具有相似含义的单词（如“国王”和“王后”）在这个 512 维空间中应该具有相似的向量表示。
            </p>
            <MathBlock formula="X_{\text{embedding}} = \text{EmbeddingLookup}(x_{\text{input}}) \times \sqrt{d_{model}}" block />
            <p className="text-sm text-slate-500">注意：在 Transformer 论文中，嵌入在加上位置编码之前会乘以 <MathBlock formula="\sqrt{d_{model}}" /> 以稳定方差。</p>
            
            <QuizSection 
              lang="zh"
              question={{
                id: 'q0',
                question: '为什么我们使用 Embedding 而不是 One-Hot 编码？',
                options: [
                  'Embedding 更容易计算。',
                  'Embedding 能捕捉语义关系，且更加紧凑（非稀疏）。',
                  'One-Hot 向量太小了。',
                  'Embedding 让我们不需要分词。'
                ],
                correctAnswer: 1,
                explanation: 'One-hot 向量非常稀疏且维度极高，并且它认为所有词之间的距离都是相等的。密集 Embedding 可以捕捉语义相似性（例如，猫和狗的向量距离很近）。'
              }} 
            />
          </div>
        )
      }
    ]
  },
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
             <h4 className="font-bold mt-6 mb-2">复杂度对比</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm text-left">
                <thead className="bg-slate-100 font-bold">
                  <tr>
                    <th className="p-2">层类型</th>
                    <th className="p-2">每层复杂度</th>
                    <th className="p-2">串行操作数</th>
                    <th className="p-2">最大路径长度</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b">
                    <td className="p-2">RNN</td>
                    <td className="p-2"><MathBlock formula="O(n \cdot d^2)" /></td>
                    <td className="p-2"><MathBlock formula="O(n)" /></td>
                    <td className="p-2"><MathBlock formula="O(n)" /></td>
                  </tr>
                  <tr>
                    <td className="p-2 bg-brand-50 font-bold text-brand-700">Self-Attention</td>
                    <td className="p-2 bg-brand-50"><MathBlock formula="O(n^2 \cdot d)" /></td>
                    <td className="p-2 bg-brand-50"><MathBlock formula="O(1)" /></td>
                    <td className="p-2 bg-brand-50"><MathBlock formula="O(1)" /></td>
                  </tr>
                </tbody>
              </table>
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
             <QuizSection 
              lang="zh"
              question={{
                id: 'q1',
                question: 'Transformer 相比 RNN 在训练上的主要优势是什么？',
                options: [
                  '它的参数更少。',
                  '它允许大规模并行化（O(1) 串行操作）。',
                  '它使用了卷积神经网络。',
                  '它不需要数据预处理。'
                ],
                correctAnswer: 1,
                explanation: '因为 Transformer 使用 Attention 一次性处理整个序列（而不是一步接一步），它可以充分利用现代 GPU 的并行计算能力。'
              }} 
            />
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
                <p className="text-sm">接收输入序列（例如：英文句子）并将其映射为包含语义信息的连续表示。它由 $N=6$ 层相同的网络堆叠而成。</p>
              </div>
              <div className="bg-white p-4 rounded shadow-sm border">
                <h4 className="font-bold text-pink-600 mb-2">Decoder (解码器 - 右侧)</h4>
                <p className="text-sm">接收编码器的输出，并逐个元素生成目标序列（例如：法文翻译）。它同样由 $N=6$ 层堆叠而成。</p>
              </div>
            </div>
             <QuizSection 
              lang="zh"
              question={{
                id: 'q2',
                question: '解码器 (Decoder) 接收什么信息？',
                options: [
                  '只有目标句子。',
                  '只有源句子。',
                  '编码器的输出 以及 目前为止生成的目标序列。',
                  '随机噪声。'
                ],
                correctAnswer: 2,
                explanation: '解码器有两个信息来源：自注意力（查看自己目前生成了什么）和 交叉注意力（查看编码器的输出）。'
              }} 
            />
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
            
            <CodeBlock code={`# 1. 缩放点积注意力 (Scaled Dot-Product Attention)
def attention(query, key, value):
    d_k = query.size(-1)
    
    # Q 乘以 K 的转置 -> 得到分数 (Scores)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 除以 sqrt(d_k) 进行缩放
    scores = scores / math.sqrt(d_k)
    
    # Softmax 归一化，得到概率分布
    attn_weights = F.softmax(scores, dim=-1)
    
    # 概率加权 V
    return torch.matmul(attn_weights, value)`} />

            <InteractiveAttention lang="zh" />
            
            <QuizSection 
              lang="zh"
              question={{
                id: 'q3-1',
                question: '在公式中，为什么要除以 sqrt(d_k)？',
                options: [
                  '为了减少计算时间。',
                  '为了防止点积过大，导致 Softmax 进入梯度极小的区域（梯度消失）。',
                  '为了让矩阵乘法成立。',
                  '这只是一个任意常数。'
                ],
                correctAnswer: 1,
                explanation: '如果点积结果很大，Softmax 的输出会接近 0 或 1，这时的梯度非常小，难以训练。缩放可以防止这种情况。'
              }} 
            />
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
             <CodeBlock code={`class MultiHeadAttention(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. 线性投影 Q, K, V
        # 将输入分割成 'h' 个头
        Q = self.w_q(x).view(batch_size, -1, self.heads, self.d_k)
        K = self.w_k(x).view(batch_size, -1, self.heads, self.d_k)
        V = self.w_v(x).view(batch_size, -1, self.heads, self.d_k)
        
        # 2. 对每个头独立计算 Attention
        # (简化代码，实际通过矩阵运算一次性完成)
        out = attention(Q, K, V)
        
        # 3. 拼接并经过线性层
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(out)`} />
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
            
            <PositionalEncodingViz lang="zh" />
          </div>
        )
      },
      {
        id: '3-4',
        title: '前馈神经网络 (FFN)',
        content: (
          <div>
             <p className="mb-4">
              除了注意力子层外，编码器和解码器的每一层都包含一个全连接的前馈网络。该网络分别且独立地应用于每个位置。
            </p>
            <p className="mb-4">
              它包含两个线性变换，中间夹一个 ReLU 激活函数。
            </p>
            <MathBlock formula="\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2" block />
            <p className="text-sm text-slate-600 mb-4">
                输入和输出的维度是 <MathBlock formula="d_{model} = 512" />，而中间层的维度是 <MathBlock formula="d_{ff} = 2048" />（维度先膨胀后压缩）。
            </p>
             <CodeBlock code={`class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        # 维度膨胀 (512 -> 2048)
        self.w_1 = nn.Linear(d_model, d_ff) 
        # 维度还原 (2048 -> 512)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # 中间使用 ReLU 激活
        return self.w_2(F.relu(self.w_1(x)))`} />
          </div>
        )
      },
      {
        id: '3-5',
        title: '编码器-解码器注意力',
        difficulty: 'advanced',
        content: (
           <div>
            <p className="mb-4">
              这是 <strong>解码器 (Decoder)</strong> 中特有的层，允许解码器查看 <strong>编码器 (Encoder)</strong> 的输出。
            </p>
            <ul className="list-disc ml-6 space-y-2 mb-4">
                <li><strong>Queries (Q):</strong> 来自解码器的前一层（我们当前正在翻译的内容）。</li>
                <li><strong>Keys (K) & Values (V):</strong> 来自编码器的输出（源语言句子）。</li>
            </ul>
            <div className="bg-purple-50 p-4 rounded-lg mb-4 border border-purple-100">
                <h4 className="font-bold text-purple-800 mb-2">为什么需要这个？</h4>
                <p className="text-sm">
                   这使得翻译结果与原始文本对齐。如果解码器试图生成“学生”这个词的法文，
                   这个机制允许它去关注编码器编码的英文单词“Student”。
                </p>
            </div>
           </div>
        )
      },
      {
        id: '3-6',
        title: '残差连接与归一化 (Add & Norm)',
        content: (
            <div>
                <p className="mb-4">
                    每个子层（Self-Attention, FFN）的输出计算公式为：
                </p>
                <MathBlock formula="\\text{LayerNorm}(x + \\text{Sublayer}(x))" block />
                <p className="mb-4">
                    <strong>残差连接 (Add)：</strong> 我们将输入 $x$ 加回到输出上。这解决了深度网络中的“梯度消失”问题。
                    <br />
                    <strong>层归一化 (Norm)：</strong> 对隐藏向量进行标准化，以稳定训练过程。
                </p>
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
            <h3 className="font-bold text-lg mb-2 text-slate-800">技术解释</h3>
            <p className="mb-4">
              <strong>填充掩码 (Padding Mask)：</strong> 忽略输入序列中为了对齐长度而填充的 0 (Padding Tokens)。
              <br/>
              <strong>前瞻掩码 (Look-Ahead Mask)：</strong> 用于解码器。在预测第 $t$ 个词时，将 $t$ 之后的位置的注意力分数设为 $-\\infty$（负无穷）。
            </p>
            
            <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-500 mt-4">
                <h4 className="font-bold mb-1 text-green-800">通俗理解：</h4>
                <p className="text-sm text-green-900">
                    这就好比在做英语填空题。
                    如果你在填第 3 个空的时候，偷看了第 4 个空的答案，那你就没有真正学会预测。
                    <strong>前瞻掩码</strong>就像是用一张纸把后面的答案挡住，强迫模型只能根据已知的上文来推断下文。
                </p>
            </div>
             <QuizSection 
              lang="zh"
              question={{
                id: 'q4-1',
                question: '为什么解码器需要前瞻掩码，而编码器不需要？',
                options: [
                  '因为编码器是双向的（能看到整个句子），而解码器是顺序生成的。',
                  '编码器不使用自注意力。',
                  '解码器速度更快。',
                  '填充 Token 只存在于解码器中。'
                ],
                correctAnswer: 0,
                explanation: '编码器一次性处理整个源句子。解码器是自回归的，意味着它每次生成一个词，不能通过“偷看”后面的词来作弊。'
              }} 
            />
          </div>
        )
      },
      {
        id: '4-2',
        title: '优化器与正则化',
        content: (
          <div>
             <h3 className="font-bold text-lg mb-2 text-slate-800">技术解释</h3>
            <p className="mb-4">
              使用 <strong>Adam</strong> 优化器 ($\beta_1=0.9, \beta_2=0.98$)。
              关键在于使用了 <strong>Warmup（热身）</strong> 策略：学习率在训练初期线性增加，随后按平方根倒数衰减。
            </p>
            <MathBlock formula="lrate = d_{\\text{model}}^{-0.5} \\cdot \\min(step\\_num^{-0.5}, step\\_num \\cdot warmup\\_steps^{-1.5})" block />
            
             <h3 className="font-bold text-lg mb-2 mt-6 text-slate-800">标签平滑 (Label Smoothing)</h3>
            <p className="mb-4">
              我们不是强迫模型对正确答案保持 100% 的自信（目标：正确词为1.0，其他为0），而是平滑目标分布。
              <br/>
              如果平滑参数 <MathBlock formula="\epsilon_{ls} = 0.1" />，正确词的概率变为 0.9，剩余的概率分配给其他词。
            </p>
            <p className="text-sm text-slate-600 mb-4">
                这虽然会增加困惑度（不确定性），但可以通过防止模型过度自信和过拟合来提高准确率和 BLEU 分数。
            </p>
          </div>
        )
      }
    ]
  },
  {
      id: 'ch5',
      title: '5. 推理与解码',
      description: '生成文本的过程',
      sections: [
          {
              id: '5-1',
              title: '自回归生成 (Auto-Regressive)',
              content: (
                  <div>
                      <p className="mb-4">
                          在推理（如翻译）阶段，模型是一个接一个地生成单词。
                      </p>
                      <ol className="list-decimal ml-6 space-y-2">
                          <li>将源句子输入编码器。</li>
                          <li>给解码器输入一个特殊的 <code>&lt;START&gt;</code> 标记。</li>
                          <li>解码器输出第一个词的概率分布。</li>
                          <li>选择概率最大的词，将其加入输入。</li>
                          <li>重复此过程，直到生成 <code>&lt;END&gt;</code> 标记。</li>
                      </ol>
                  </div>
              )
          },
          {
              id: '5-2',
              title: '贪婪搜索与集束搜索',
              content: (
                  <div>
                      <p className="mb-4">
                          <strong>贪婪搜索 (Greedy Search)：</strong> 每一步都选择概率最高的那个词。速度快，但容易陷入局部最优（生成的句子可能不通顺）。
                      </p>
                      <p className="mb-4">
                          <strong>集束搜索 (Beam Search)：</strong> 每一步都保留前 $k$ 个（Beam Width）最可能的句子片段。这允许模型探索多种可能性，从而找到全局更好的翻译。
                      </p>
                      <QuizSection 
                        lang="zh"
                        question={{
                          id: 'q5',
                          question: '相比贪婪搜索，集束搜索 (Beam Search) 的主要优势是什么？',
                          options: [
                            '它更快。',
                            '它探索多条潜在的句子路径，避免陷入局部最优。',
                            '它占用更少内存。',
                            '它不需要解码器。'
                          ],
                          correctAnswer: 1,
                          explanation: '贪婪搜索可能现在选了一个概率最高的词，但导致后面无路可走。集束搜索通过保留多个候选，让“眼光放得更长远”。'
                        }} 
                      />
                  </div>
              )
          }
      ]
  },
  {
      id: 'ch6',
      title: '6. Transformer 家族',
      description: '架构的演变',
      sections: [
          {
              id: '6-1',
              title: '仅编码器 (BERT)',
              content: (
                  <div>
                      <h4 className="font-bold text-slate-800">BERT (Bidirectional Encoder Representations from Transformers)</h4>
                      <p className="text-sm mb-2">
                          只使用了 <strong>Encoder</strong> 堆叠。
                      </p>
                      <p className="text-sm mb-2">
                          <strong>目标：</strong> 理解文本。它同时查看上下文（双向）。
                      </p>
                      <p className="text-sm">
                          <strong>任务：</strong> 文本分类、情感分析、命名实体识别。
                      </p>
                  </div>
              )
          },
          {
              id: '6-2',
              title: '仅解码器 (GPT)',
              content: (
                  <div>
                      <h4 className="font-bold text-slate-800">GPT (Generative Pre-trained Transformer)</h4>
                      <p className="text-sm mb-2">
                          只使用了 <strong>Decoder</strong> 堆叠（带有掩码的自注意力）。
                      </p>
                      <p className="text-sm mb-2">
                          <strong>目标：</strong> 生成文本。它根据前面的词预测下一个词。
                      </p>
                      <p className="text-sm">
                          <strong>任务：</strong> 文本生成、聊天机器人、代码补全。
                      </p>
                  </div>
              )
          },
          {
              id: '6-3',
              title: '编码器-解码器 (T5 / BART)',
              content: (
                  <div>
                      <h4 className="font-bold text-slate-800">T5 (Text-to-Text Transfer Transformer)</h4>
                      <p className="text-sm mb-2">
                          使用了完整的原始架构。
                      </p>
                      <p className="text-sm">
                          <strong>任务：</strong> 翻译、摘要（序列到序列任务）。
                      </p>
                  </div>
              )
          }
      ]
  },
  {
    id: 'ch7',
    title: '7. 实验结果与影响',
    description: '它真的有效吗？',
    sections: [
      {
        id: '7-1',
        title: '多维度性能分析',
        content: (
          <div>
             <p className="mb-4">
                Transformer 在 WMT 2014 英德和英法翻译任务上均达到了当时的 SOTA (State-of-the-art) 水平。
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
                    <h4 className="font-bold text-green-600">训练成本 (Cost)</h4>
                    <p className="text-sm mt-1">
                        Transformer (Base) 仅消耗了 <strong>$3.3 \cdot 10^{18}$</strong> 次浮点运算。
                        Transformer (Big) 在 8 张 P100 GPU 上仅训练了 <strong>3.5 天</strong>。相比之下，之前的模型往往需要训练数周。
                    </p>
                </div>
                <div className="p-4 border rounded bg-white">
                    <h4 className="font-bold text-purple-600">泛化能力 (Generalization)</h4>
                    <p className="text-sm mt-1">
                        论文证明了模型可以很好地迁移到其他任务。例如在 <strong>英语成分句法分析 (English Constituency Parsing)</strong> 任务中，它在几乎没有针对性调优的情况下也取得了极好的成绩。
                    </p>
                </div>
            </div>
          </div>
        )
      }
    ]
  },
  {
    id: 'ch8',
    title: '8. 资源与代码',
    description: '深入研究',
    sections: [
      {
        id: '8-1',
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
                            <div className="font-bold text-slate-800 group-hover:underline">Tensor2Tensor (GitHub)</div>
                            <div className="text-sm text-slate-500">论文使用的官方 TensorFlow 实现。</div>
                        </div>
                    </a>
                </li>
                 <li>
                    <a href="https://github.com/pytorch/fairseq" target="_blank" rel="noreferrer" className="flex items-center gap-2 p-4 border rounded hover:bg-slate-50 transition group">
                        <span className="text-2xl">🔥</span>
                        <div>
                            <div className="font-bold text-slate-800 group-hover:underline">PyTorch FairSeq (GitHub)</div>
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