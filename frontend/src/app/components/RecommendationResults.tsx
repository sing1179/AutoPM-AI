import { Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';

const API_URL = import.meta.env.VITE_API_URL || '';

interface RecommendationResultsProps {
  query: string;
  files: File[];
}

export function RecommendationResults({ query, files }: RecommendationResultsProps) {
  const [recommendations, setRecommendations] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRecommendations = async () => {
      if (files.length === 0) return;

      setLoading(true);
      setError(null);

      try {
        const formData = new FormData();
        formData.append('query', query);
        files.forEach((file) => formData.append('files', file));

        const res = await fetch(`${API_URL || ''}/api/recommendations`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail || 'Failed to get recommendations');
        }

        const data = await res.json();
        setRecommendations(data.recommendations);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, [query, files]);

  if (loading) {
    return (
      <div className="w-full flex items-center justify-center py-12">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-8 h-8 text-white/60 animate-spin" />
          <p className="text-white/50 text-sm">Analyzing your documents...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full p-4 rounded-xl bg-red-500/10 border border-red-500/20">
        <p className="text-red-400 text-sm">{error}</p>
        <p className="text-white/40 text-xs mt-2">Ensure GROQ_API_KEY is set in the API server.</p>
      </div>
    );
  }

  if (!recommendations) return null;

  return (
    <div className="w-full space-y-4 animate-in fade-in duration-500">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-white/90 font-medium">AI Recommendations</h3>
          <p className="text-white/50 text-sm mt-1">
            Analyzed {files.length} document{files.length !== 1 ? 's' : ''}
          </p>
        </div>
      </div>

      <div className="p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10 prose prose-invert prose-sm max-w-none">
        <ReactMarkdown
          components={{
            h1: ({ children }) => <h1 className="text-white font-semibold text-lg mb-2">{children}</h1>,
            h2: ({ children }) => <h2 className="text-white font-semibold text-base mt-4 mb-2">{children}</h2>,
            h3: ({ children }) => <h3 className="text-white/90 font-medium text-sm mt-3 mb-1">{children}</h3>,
            p: ({ children }) => <p className="text-white/70 text-sm leading-relaxed mb-2">{children}</p>,
            ul: ({ children }) => <ul className="list-disc list-inside text-white/70 text-sm space-y-1 mb-2">{children}</ul>,
            ol: ({ children }) => <ol className="list-decimal list-inside text-white/70 text-sm space-y-1 mb-2">{children}</ol>,
            li: ({ children }) => <li className="text-white/70">{children}</li>,
            strong: ({ children }) => <strong className="text-white font-medium">{children}</strong>,
          }}
        >
          {recommendations}
        </ReactMarkdown>
      </div>
    </div>
  );
}
