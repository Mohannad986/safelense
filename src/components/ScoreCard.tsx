import React from 'react';
import { AlertTriangle, CheckCircle, Eye, Volume2, Zap, Shield } from 'lucide-react';

interface ScoreCardProps {
  title: string;
  score: number;
  type: 'overall' | 'frame' | 'audio' | 'sync';
  description: string;
}

const ScoreCard: React.FC<ScoreCardProps> = ({ title, score, type, description }) => {
  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getScoreBarColor = (score: number) => {
    if (score >= 80) return 'bg-green-400';
    if (score >= 60) return 'bg-yellow-400';
    return 'bg-red-400';
  };

  const getScoreBackground = (score: number) => {
    if (score >= 80) return 'bg-green-500/20 border-green-400/30';
    if (score >= 60) return 'bg-yellow-500/20 border-yellow-400/30';
    return 'bg-red-500/20 border-red-400/30';
  };

  const getIcon = () => {
    switch (type) {
      case 'overall':
        return <Shield className="w-8 h-8" />;
      case 'frame':
        return <Eye className="w-8 h-8" />;
      case 'audio':
        return <Volume2 className="w-8 h-8" />;
      case 'sync':
        return <Zap className="w-8 h-8" />;
      default:
        return <Shield className="w-8 h-8" />;
    }
  };

  const getStatusIcon = () => {
    if (score >= 70) {
      return <CheckCircle className="w-5 h-5 text-green-400" />;
    }
    return <AlertTriangle className="w-5 h-5 text-red-400" />;
  };

  const getStatusText = () => {
    if (score >= 80) return 'Authentic';
    if (score >= 60) return 'Suspicious';
    return 'High Risk';
  };

  return (
    <div className={`rounded-xl p-6 border backdrop-blur-sm transition-all duration-300 hover:scale-105 ${getScoreBackground(score)}`}>
      <div className="flex items-center justify-between mb-4">
        <div className={getScoreColor(score)}>
          {getIcon()}
        </div>
        {getStatusIcon()}
      </div>
      
      <h3 className="text-lg font-semibold text-white mb-1">{title}</h3>
      <p className="text-sm text-gray-300 mb-4">{description}</p>
      
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className={`text-3xl font-bold ${getScoreColor(score)}`}>
            {score}%
          </span>
          <span className={`text-sm font-medium ${getScoreColor(score)}`}>
            {getStatusText()}
          </span>
        </div>
        
        <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-1000 ease-out ${getScoreBarColor(score)}`}
            style={{ width: `${score}%` }}
          ></div>
        </div>
      </div>
      
      {type === 'overall' && score < 70 && (
        <div className="mt-4 p-2 bg-red-500/20 rounded-lg">
          <p className="text-xs text-red-300">
            ⚠️ Media authenticity questionable
          </p>
        </div>
      )}
    </div>
  );
};

export default ScoreCard;