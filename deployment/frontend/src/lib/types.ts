export interface Sequence {
  name: string;
  fasta: string;
}

export interface InputPanelProps {
  onSubmit: (sequences: string) => void;
  isLoading?: boolean;
}

export interface WindowResult {
  sequence: string;
  position: number;
  model_prediction?: number;
  fullSequence?: string;
  error?: string;
}

export interface PositionResult {
  position: number;
  score: number;
  confidence?: number;
  isUpdated?: boolean;
}

export interface SequenceAnalysis {
  sequence: string;
  results: PositionResult[];
  isComplete: boolean;
  error?: string;
}

export interface ResultType {
  sequence: string;
  results: PositionResult[];
  error: string | null;
  isLoading: boolean;
  progress?: {
    position: number;
    totalLength: number;
  };
} 