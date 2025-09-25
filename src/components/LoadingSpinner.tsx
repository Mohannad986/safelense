import React, { useState, useEffect } from 'react';
import { Brain, Eye, Volume2, Zap } from 'lucide-react';

interface LoadingSpinnerProps {
  filename: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ filename }) => {
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    { icon: Eye, label: 'Extracting video frames', description: 'Analyzing visual content...' },
    { icon: Volume2, label: 'Processing audio track', description: 'Detecting audio patterns...' },
    { icon: Brain, label: 'Running deepfake detection', description: 'AI model analyzing frames...' },
    { icon: Zap, label: 'Computing final scores', description: 'Generating authenticity report...' },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev + 1) % steps.length);
    }, 1500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20 text-center">
      <div className="mb-6">
        <div className="w-16 h-16 mx-auto mb-4 relative">
          <div className="absolute inset-0 border-4 border-purple-400/30 rounded-full"></div>
          <div className="absolute inset-0 border-4 border-transparent border-t-purple-400 rounded-full animate-spin"></div>
        </div>
        
        <h3 className="text-xl font-semibold text-white mb-2">Analyzing Video</h3>
        <p className="text-gray-300 text-sm mb-6">Processing: {filename}</p>
      </div>

      <div className="space-y-4">
        {steps.map((step, index) => {
          const Icon = step.icon;
          const isActive = index === currentStep;
          const isComplete = index < currentStep;
          
          return (
            <div
              key={index}
              className={`flex items-center gap-4 p-3 rounded-lg transition-all duration-300 ${
                isActive 
                  ? 'bg-purple-500/30 border border-purple-400/50 scale-105' 
                  : isComplete
                    ? 'bg-green-500/20 border border-green-400/30'
                    : 'bg-white/5 border border-white/10'
              }`}
            >
              <div className={`p-2 rounded-full ${
                isActive 
                  ? 'bg-purple-400 text-white' 
                  : isComplete
                    ? 'bg-green-400 text-white'
                    : 'bg-gray-600 text-gray-300'
              }`}>
                <Icon className="w-4 h-4" />
              </div>
              
              <div className="flex-1 text-left">
                <div className={`font-medium ${
                  isActive ? 'text-purple-300' : isComplete ? 'text-green-300' : 'text-gray-400'
                }`}>
                  {step.label}
                </div>
                <div className="text-sm text-gray-400">
                  {step.description}
                </div>
              </div>
              
              {isActive && (
                <div className="flex space-x-1">
                  <div className="w-1 h-1 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-1 h-1 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-1 h-1 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              )}
              
              {isComplete && (
                <div className="text-green-400">✓</div>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-6 text-sm text-gray-400">
        This may take a few moments depending on video length
      </div>
    </div>
  );
};

export default LoadingSpinner;