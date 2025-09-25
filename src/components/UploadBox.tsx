import React, { useCallback, useState } from 'react';
import { Upload, Video, AlertCircle } from 'lucide-react';

interface UploadBoxProps {
  onFileUpload: (file: File) => void;
}

const UploadBox: React.FC<UploadBoxProps> = ({ onFileUpload }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string>('');

  const validateFile = (file: File): boolean => {
    const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/webm', 'video/mkv'];
    const maxSize = 100 * 1024 * 1024; // 100MB

    if (!validTypes.includes(file.type)) {
      setError('Please upload a valid video file (MP4, AVI, MOV, WebM, MKV)');
      return false;
    }

    if (file.size > maxSize) {
      setError('File size must be less than 100MB');
      return false;
    }

    setError('');
    return true;
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      const file = files[0];
      if (validateFile(file)) {
        onFileUpload(file);
      }
    }
  }, [onFileUpload]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (validateFile(file)) {
        onFileUpload(file);
      }
    }
  }, [onFileUpload]);

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${
          isDragOver
            ? 'border-purple-400 bg-purple-500/20 scale-105'
            : 'border-white/30 bg-white/10 hover:border-purple-400/60 hover:bg-white/15'
        } backdrop-blur-sm`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="video/*"
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          id="video-upload"
        />
        
        <div className="space-y-4">
          <div className={`transition-all duration-300 ${isDragOver ? 'scale-110' : ''}`}>
            <Video className="w-16 h-16 mx-auto text-purple-400 mb-4" />
          </div>
          
          <div>
            <h3 className="text-xl font-semibold text-white mb-2">
              Upload Video for Analysis
            </h3>
            <p className="text-gray-300 mb-6">
              Drag and drop your video file here, or click to browse
            </p>
          </div>
          
          <label
            htmlFor="video-upload"
            className="inline-flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white px-8 py-3 rounded-lg font-medium transition-all duration-200 hover:scale-105 cursor-pointer"
          >
            <Upload className="w-5 h-5" />
            Choose File
          </label>
          
          <p className="text-sm text-gray-400 mt-4">
            Supported formats: MP4, AVI, MOV, WebM, MKV (max 100MB)
          </p>
        </div>
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}
    </div>
  );
};

export default UploadBox;