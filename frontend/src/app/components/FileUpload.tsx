import { Upload, FileText, X, FolderOpen } from 'lucide-react';
import { useState, useRef } from 'react';

interface FileUploadProps {
  onFilesChange: (files: File[]) => void;
  uploadedFiles: File[];
}

export function FileUpload({ onFilesChange, uploadedFiles }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    onFilesChange([...uploadedFiles, ...files]);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      onFilesChange([...uploadedFiles, ...files]);
    }
  };

  const handleRemoveFile = (index: number) => {
    const newFiles = uploadedFiles.filter((_, i) => i !== index);
    onFilesChange(newFiles);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="flex items-start gap-3">
      {/* Quick action chips */}
      <button
        onClick={() => fileInputRef.current?.click()}
        className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 
                 border border-white/10 text-white/70 hover:bg-white/10 hover:text-white
                 transition-all text-sm flex-shrink-0"
      >
        <FolderOpen className="w-4 h-4" />
        <span>Upload documents</span>
      </button>

      {uploadedFiles.length > 0 && (
        <div className="flex gap-2 flex-wrap flex-1">
          {uploadedFiles.map((file, index) => (
            <div
              key={index}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 backdrop-blur-sm 
                       border border-white/10 group hover:bg-white/10 transition-all"
            >
              <FileText className="w-3.5 h-3.5 text-white/50 flex-shrink-0" />
              <span className="text-white/80 text-xs truncate max-w-[120px]">{file.name}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemoveFile(index);
                }}
                className="p-0.5 rounded hover:bg-white/10 transition-all"
              >
                <X className="w-3 h-3 text-white/50" />
              </button>
            </div>
          ))}
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        multiple
        onChange={handleFileSelect}
        className="hidden"
        accept=".pdf,.doc,.docx,.txt,.md,.csv,.xlsx,.xls"
      />
    </div>
  );
}