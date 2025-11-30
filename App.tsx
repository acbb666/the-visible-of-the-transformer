import React, { useState } from 'react';
import { Menu, BookOpen, ChevronRight, BrainCircuit, Languages } from 'lucide-react';
import { getChapters, uiTranslations } from './data';
import GeminiTutor from './components/GeminiTutor';
import { Language } from './types';

const App: React.FC = () => {
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const [activeChapterId, setActiveChapterId] = useState<string | null>(null);
  const [lang, setLang] = useState<Language>('zh');

  const chapters = getChapters(lang);
  const activeChapter = chapters.find(c => c.id === activeChapterId);
  const t = uiTranslations[lang];

  const toggleLang = () => {
    setLang(prev => prev === 'en' ? 'zh' : 'en');
  };

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-slate-50 text-slate-900">
      
      {/* Mobile Header */}
      <div className="md:hidden bg-white border-b p-4 flex items-center justify-between sticky top-0 z-40">
        <div className="flex items-center gap-2 font-bold text-lg text-brand-700">
          <BrainCircuit size={24} />
          <span>Transformer Explained</span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={toggleLang} className="p-2 text-slate-600 font-bold flex items-center gap-1">
             <Languages size={20} />
             <span className="text-sm">{lang.toUpperCase()}</span>
          </button>
          <button onClick={() => setSidebarOpen(!isSidebarOpen)} className="p-2 text-slate-600">
            <Menu size={24} />
          </button>
        </div>
      </div>

      {/* Sidebar Navigation */}
      <aside className={`
        fixed inset-y-0 left-0 z-30 w-72 bg-white border-r border-slate-200 transform transition-transform duration-300 ease-in-out
        md:relative md:translate-x-0
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="p-6 border-b border-slate-100 hidden md:flex items-center justify-between font-bold text-xl text-brand-700">
          <div className="flex items-center gap-2">
             <BrainCircuit size={28} />
             <span>Transformer</span>
          </div>
        </div>

        <nav className="p-4 space-y-1 overflow-y-auto h-[calc(100vh-140px)] md:h-[calc(100vh-80px)]">
          <button
            onClick={() => { setActiveChapterId(null); setSidebarOpen(false); }}
            className={`w-full text-left px-4 py-3 rounded-lg flex items-center gap-3 transition-colors ${
              activeChapterId === null ? 'bg-brand-50 text-brand-700 font-medium' : 'text-slate-600 hover:bg-slate-50'
            }`}
          >
            <BookOpen size={18} />
            <span>{t.home}</span>
          </button>

          <div className="pt-4 pb-2 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
            {t.curriculum}
          </div>

          {chapters.map((chapter) => (
            <button
              key={chapter.id}
              onClick={() => { setActiveChapterId(chapter.id); setSidebarOpen(false); }}
              className={`w-full text-left px-4 py-3 rounded-lg flex items-start gap-3 transition-colors group ${
                activeChapterId === chapter.id ? 'bg-brand-50 text-brand-700' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <span className={`mt-0.5 w-6 h-6 rounded-full flex items-center justify-center text-xs flex-shrink-0 ${
                activeChapterId === chapter.id ? 'bg-brand-200 text-brand-800' : 'bg-slate-100 text-slate-500 group-hover:bg-slate-200'
              }`}>
                {chapter.id.replace('ch', '')}
              </span>
              <span className={`text-sm ${activeChapterId === chapter.id ? 'font-medium' : ''}`}>
                {chapter.title.split('. ')[1] || chapter.title}
              </span>
            </button>
          ))}
        </nav>
        
        {/* Desktop Language Switcher at Bottom */}
        <div className="hidden md:block absolute bottom-0 w-full p-4 border-t border-slate-100 bg-white">
           <button 
             onClick={toggleLang}
             className="flex items-center justify-center gap-2 w-full py-2 bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700 font-medium transition-colors"
           >
             <Languages size={18} />
             <span>Switch to {lang === 'en' ? '中文' : 'English'}</span>
           </button>
        </div>
      </aside>

      {/* Overlay for mobile sidebar */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/20 z-20 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto h-screen scroll-smooth">
        <div className="max-w-4xl mx-auto px-4 py-12 md:px-12">
          
          {activeChapter ? (
            // Chapter View
            <div className="animate-in fade-in duration-500">
              <header className="mb-10 border-b pb-6">
                <div className="flex items-center gap-2 text-sm text-brand-600 font-medium mb-2">
                  <span>{lang === 'en' ? 'Chapter' : '第'} {activeChapter.id.replace('ch', '')} {lang === 'zh' && '章'}</span>
                  <ChevronRight size={14} />
                  <span>{t.curriculum}</span>
                </div>
                <h1 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">{activeChapter.title}</h1>
                <p className="text-xl text-slate-500">{activeChapter.description}</p>
              </header>

              <div className="space-y-16">
                {activeChapter.sections.map((section) => (
                  <section key={section.id} id={section.id} className="scroll-mt-20">
                    <div className="flex items-center gap-3 mb-4">
                      <h2 className="text-2xl font-bold text-slate-800">{section.title}</h2>
                      {section.difficulty && (
                        <span className={`px-2 py-1 rounded text-xs font-semibold uppercase tracking-wide
                          ${section.difficulty === 'basic' ? 'bg-green-100 text-green-700' : ''}
                          ${section.difficulty === 'intermediate' ? 'bg-yellow-100 text-yellow-700' : ''}
                          ${section.difficulty === 'advanced' ? 'bg-red-100 text-red-700' : ''}
                        `}>
                          {section.difficulty}
                        </span>
                      )}
                    </div>
                    <div className="prose prose-slate max-w-none text-slate-600 leading-relaxed">
                      {section.content}
                    </div>
                  </section>
                ))}
              </div>

              {/* Navigation Footer */}
              <div className="mt-20 pt-8 border-t flex justify-between">
                 {(() => {
                   const idx = chapters.findIndex(c => c.id === activeChapter.id);
                   const prev = chapters[idx - 1];
                   const next = chapters[idx + 1];
                   return (
                     <>
                       {prev ? (
                         <button onClick={() => setActiveChapterId(prev.id)} className="text-left group">
                           <div className="text-xs text-slate-400 mb-1">{t.prev}</div>
                           <div className="font-semibold text-brand-600 group-hover:underline">{prev.title}</div>
                         </button>
                       ) : <div />}
                       
                       {next ? (
                         <button onClick={() => setActiveChapterId(next.id)} className="text-right group">
                           <div className="text-xs text-slate-400 mb-1">{t.next}</div>
                           <div className="font-semibold text-brand-600 group-hover:underline">{next.title}</div>
                         </button>
                       ) : <div />}
                     </>
                   );
                 })()}
              </div>
            </div>
          ) : (
            // Home View
            <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
              <div className="text-center py-10">
                <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900 mb-6">
                  {t.title.split(' ')[0]} <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-500 to-purple-600">{t.subtitle.split(' ')[1] || 'Decoded'}</span>
                </h1>
                <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
                  {t.desc}
                </p>
                
                <div className="mt-10 flex flex-wrap justify-center gap-4">
                  <button 
                    onClick={() => setActiveChapterId('ch1')}
                    className="px-8 py-3 bg-brand-600 text-white font-bold rounded-full shadow-lg hover:bg-brand-700 transition transform hover:-translate-y-1"
                  >
                    {t.start}
                  </button>
                  <a 
                    href="https://arxiv.org/abs/1706.03762" 
                    target="_blank" 
                    rel="noreferrer"
                    className="px-8 py-3 bg-white text-slate-700 font-bold rounded-full shadow border border-slate-200 hover:bg-slate-50 transition"
                  >
                    {t.paper}
                  </a>
                </div>
              </div>

              <div className="grid md:grid-cols-3 gap-6">
                {[
                  { label: t.stats.published, value: "NeurIPS 2017", icon: "📅" },
                  { label: t.stats.citations, value: "100,000+", icon: "🎓" },
                  { label: t.stats.concept, value: "Self-Attention", icon: "🔑" }
                ].map((stat, i) => (
                  <div key={i} className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 text-center">
                    <div className="text-2xl mb-2">{stat.icon}</div>
                    <div className="text-sm text-slate-500 uppercase tracking-wide font-semibold">{stat.label}</div>
                    <div className="text-xl font-bold text-slate-800">{stat.value}</div>
                  </div>
                ))}
              </div>

              <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-8 text-white relative overflow-hidden">
                <div className="relative z-10">
                  <h3 className="text-2xl font-bold mb-4">{t.whyTitle}</h3>
                  <p className="text-slate-300 mb-6 max-w-xl">
                    {t.whyDesc}
                  </p>
                </div>
                {/* Abstract background element */}
                <div className="absolute top-0 right-0 -mt-10 -mr-10 w-64 h-64 bg-brand-500/20 rounded-full blur-3xl"></div>
              </div>
            </div>
          )}
        </div>
      </main>
      
      <GeminiTutor lang={lang} />
    </div>
  );
};

export default App;