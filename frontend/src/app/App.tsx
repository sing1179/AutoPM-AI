import { useState } from 'react';
import { FileUpload } from './components/FileUpload';
import { RecommendationInput } from './components/RecommendationInput';
import { RecommendationResults } from './components/RecommendationResults';
import { Logo } from './components/Logo';
import { MessageSquare } from 'lucide-react';

export default function App() {
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [query, setQuery] = useState<string>('');
  const [showResults, setShowResults] = useState(false);
  const [isInteracting, setIsInteracting] = useState(false);

  const handleQuerySubmit = (submittedQuery: string) => {
    setQuery(submittedQuery);
    setShowResults(true);
    setIsInteracting(true);
  };

  return (
    <div className="size-full min-h-screen relative overflow-hidden bg-black">
      {/* Dotted grid pattern */}
      <div className="absolute inset-0" style={{
        backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px)',
        backgroundSize: '40px 40px'
      }} />

      {/* Large glowing purple orb */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] 
                      bg-amber-500/40 rounded-full blur-[120px] animate-pulse" 
           style={{ animationDuration: '4s' }} />
      
      {/* Secondary pink glow */}
      <div className="absolute bottom-0 right-0 w-[600px] h-[600px] 
                      bg-orange-400/30 rounded-full blur-[100px]" />

      {/* Navigation */}
      <nav className="relative z-20 flex items-center justify-between px-8 py-6">
        <div className="flex items-center gap-3">
          <Logo />
          <span className="text-white text-lg font-semibold">AutoPM-AI</span>
        </div>
        <div className="flex items-center gap-8">
          <button className="text-white/70 hover:text-white transition-colors text-sm">Research notes</button>
          <button className="px-4 py-2 rounded-lg border border-white/20 text-white text-sm 
                           hover:bg-white/5 transition-all">Login</button>
          <button className="px-4 py-2 rounded-lg bg-white text-black text-sm font-medium
                           hover:bg-white/90 transition-all">Create account</button>
        </div>
      </nav>

      {/* Main content */}
      <div className="relative z-10 flex flex-col items-center justify-center px-6 pt-20 pb-12">
        <div className="w-full max-w-4xl space-y-12">
          {/* Hero section - hidden when interacting */}
          {!isInteracting && (
            <div className="text-center space-y-6">
              <h1 className="text-6xl font-bold text-white tracking-tight leading-tight">
                Optimized for Thought
                <br />
                Built for Action
              </h1>
              
              <p className="text-white/60 text-lg">
                Think smarter and act faster, from idea to execution in seconds.
              </p>
            </div>
          )}

          {/* Main glass card - centered initially, moves to bottom when interacting */}
          <div className={`rounded-3xl bg-black/40 backdrop-blur-xl border border-white/10 shadow-2xl 
                        overflow-hidden relative transition-all duration-500 ${
                          isInteracting ? 'fixed bottom-8 left-1/2 -translate-x-1/2 w-full max-w-4xl' : ''
                        }`}>
            <div className="p-8 space-y-6">
              <RecommendationInput 
                onSubmit={handleQuerySubmit}
                isProcessing={false}
              />

              {/* Upload and Feedback buttons on same line */}
              <div className="flex items-center justify-between gap-4">
                <FileUpload 
                  uploadedFiles={uploadedFiles}
                  onFilesChange={setUploadedFiles}
                />
                <button className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 
                                 border border-white/10 text-white/70 hover:bg-white/10 hover:text-white
                                 transition-all text-sm flex-shrink-0">
                  <MessageSquare className="w-4 h-4" />
                  <span>Feedback</span>
                </button>
              </div>

              {showResults && uploadedFiles.length > 0 && !isInteracting && (
                <div className="pt-4">
                  <RecommendationResults 
                    query={query}
                    files={uploadedFiles}
                  />
                </div>
              )}

              {showResults && uploadedFiles.length === 0 && !isInteracting && (
                <div className="text-center py-8">
                  <p className="text-white/40 text-sm">Upload documents to get AI-powered recommendations</p>
                </div>
              )}
            </div>
          </div>

          {/* Results area when interacting - shows above the fixed input */}
          {isInteracting && showResults && uploadedFiles.length > 0 && (
            <div className="mb-64">
              <RecommendationResults 
                query={query}
                files={uploadedFiles}
              />
            </div>
          )}

          {/* Trusted by section - hidden when interacting */}
          {!isInteracting && (
            <div className="text-center space-y-6 pt-8">
              <p className="text-white/40 text-sm">Trusted by</p>
              <div className="flex items-center justify-center gap-12 opacity-40">
                <span className="text-white text-lg font-semibold">Atlassian</span>
                <span className="text-white text-lg font-semibold">Notion</span>
                <span className="text-white text-lg font-semibold">Linear</span>
                <span className="text-white text-lg font-semibold">GitHub</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Help button - bottom right */}
      <button
        className="fixed bottom-6 right-6 w-10 h-10 rounded-full bg-white/10 border border-white/10 
                   flex items-center justify-center text-white/70 hover:bg-white/20 hover:text-white
                   transition-all z-50 text-lg font-medium"
        aria-label="Help"
      >
        ?
      </button>
    </div>
  );
}