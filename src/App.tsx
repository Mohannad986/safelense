import React, { useState } from 'react';
import { Shield, Upload, BarChart3, Download } from 'lucide-react';
import UploadBox from './components/UploadBox';
import LoadingSpinner from './components/LoadingSpinner';
import ScoreCard from './components/ScoreCard';
import Header from './components/Header';

interface AnalysisResults {
  frame_fake_score: number;
  audio_fake_score: number;
  av_sync_score: number;
  overall_authenticity: number;
  filename: string;
  analysis_time: string;
}

function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [error, setError] = useState<string>('');

  const handleFileUpload = async (file: File) => {
    setUploadedFile(file);
    setIsAnalyzing(true);
    setResults(null);
    setError('');

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', file);

      // Send to backend API
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const analysisResults = await response.json();
      setResults(analysisResults);
      
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err instanceof Error ? err.message : 'Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const downloadReport = () => {
    if (!results) return;
    
    const report = {
      ...results,
      analysis_details: {
        frame_analysis: "Deep learning model detected visual inconsistencies in facial features",
        audio_analysis: "Audio spoofing detection using AASIST model",
        lipsync_analysis: "Audiovisual alignment computed using MediaPipe landmarks",
        model_versions: {
          deepfake_detector: "prithivMLmods/deepfake-detector-model-v1",
          audio_spoof_detector: "MTUCI/AASIST3",
          lipsync_detector: "MediaPipe v0.10.9"
        }
      }
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trustai_report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const resetAnalysis = () => {
    setResults(null);
    setUploadedFile(null);
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {!uploadedFile && !isAnalyzing && !results && (
          <div className="max-w-4xl mx-auto text-center mb-12">
            <div className="mb-8">
              <Shield className="w-20 h-20 mx-auto mb-6 text-purple-400" />
              <h1 className="text-5xl font-bold text-white mb-6">
                Trust<span className="text-purple-400">AI</span>
              </h1>
              <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
                Advanced AI-powered detection for deepfake videos, manipulated audio, and synthetic media. 
                Ensure media authenticity with cutting-edge machine learning.
              </p>
            </div>
            
            <UploadBox onFileUpload={handleFileUpload} />
            
            {error && (
              <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg">
                <p className="text-red-300">{error}</p>
                <p className="text-red-400 text-sm mt-2">
                  Make sure the backend server is running on port 8000
                </p>
              </div>
            )}
            
            <div className="grid md:grid-cols-3 gap-6 mt-12">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <BarChart3 className="w-8 h-8 text-purple-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">Frame Analysis</h3>
                <p className="text-gray-300 text-sm">
                  Deep learning detection of facial inconsistencies and visual artifacts
                </p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <Upload className="w-8 h-8 text-purple-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">Audio Spoofing</h3>
                <p className="text-gray-300 text-sm">
                  Advanced detection of synthetic speech and voice cloning
                </p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <Shield className="w-8 h-8 text-purple-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">Lip-Sync Analysis</h3>
                <p className="text-gray-300 text-sm">
                  Audiovisual alignment detection using facial landmark tracking
                </p>
              </div>
            </div>
          </div>
        )}

        {isAnalyzing && (
          <div className="max-w-2xl mx-auto">
            <LoadingSpinner filename={uploadedFile?.name || ''} />
          </div>
        )}

        {results && (
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-2">Analysis Complete</h2>
              <p className="text-gray-300">Results for: {results.filename}</p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <ScoreCard
                title="Overall Authenticity"
                score={results.overall_authenticity}
                type="overall"
                description="Combined authenticity score"
              />
              <ScoreCard
                title="Frame Analysis"
                score={results.frame_fake_score}
                type="frame"
                description="Visual deepfake detection"
              />
              <ScoreCard
                title="Audio Analysis"
                score={results.audio_fake_score}
                type="audio"
                description="Synthetic audio detection"
              />
              <ScoreCard
                title="Lip-Sync Score"
                score={results.av_sync_score}
                type="sync"
                description="Audiovisual alignment"
              />
            </div>

            <div className="text-center space-y-4">
              <button
                onClick={downloadReport}
                className="inline-flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg font-medium transition-all duration-200 hover:scale-105"
              >
                <Download className="w-5 h-5" />
                Download Report
              </button>
              
              <div>
                <button
                  onClick={resetAnalysis}
                  className="text-purple-400 hover:text-purple-300 underline"
                >
                  Analyze Another Video
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;