import { useState } from 'react';
import { Logo } from './Logo';

interface RecommendationInputProps {
  onSubmit: (query: string) => void;
  isProcessing?: boolean;
}

export function RecommendationInput({ onSubmit, isProcessing = false }: RecommendationInputProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSubmit(query);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative">
        <div className="absolute left-5 top-5 pointer-events-none">
          <Logo className="w-5 h-5" />
        </div>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask for anything or use a command"
          rows={4}
          className="w-full pl-14 pr-4 py-4 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 
                     text-white placeholder:text-white/40 focus:outline-none focus:border-white/20 
                     focus:bg-white/10 transition-all duration-200 resize-none"
        />
        <button
          type="submit"
          disabled={!query.trim() || isProcessing}
          className="absolute bottom-4 right-4 px-5 py-2 rounded-lg bg-white/10 border border-white/10
                     hover:bg-white/20 disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all duration-200 text-white text-sm font-medium"
        >
          Generate
        </button>
      </div>
    </form>
  );
}