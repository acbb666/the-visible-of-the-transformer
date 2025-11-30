import React, { useEffect, useRef } from 'react';

interface MathBlockProps {
  formula: string;
  block?: boolean;
}

const MathBlock: React.FC<MathBlockProps> = ({ formula, block = false }) => {
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (containerRef.current && (window as any).katex) {
      try {
        (window as any).katex.render(formula, containerRef.current, {
          throwOnError: false,
          displayMode: block,
        });
      } catch (e) {
        console.error("KaTeX render error", e);
        containerRef.current.innerText = formula;
      }
    } else if (containerRef.current) {
        containerRef.current.innerText = formula; 
    }
  }, [formula, block]);

  return <span ref={containerRef} className={`${block ? "my-4 block text-center" : "mx-1"}`} />;
};

export default MathBlock;