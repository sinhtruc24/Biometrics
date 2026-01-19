import { Card } from "./ui/card";
import { Progress } from "./ui/progress";
import { Badge } from "./ui/badge";
import { CheckCircle2, XCircle } from "lucide-react";

interface ResultPanelProps {
  similarity: number;
  isSamePerson: boolean;
  inferenceTime: number;
  threshold: number;
  isVisible: boolean;
}

export function ResultPanel({ similarity, isSamePerson, inferenceTime, threshold, isVisible }: ResultPanelProps) {
  if (!isVisible) return null;

  return (
    <Card className="p-4 bg-gradient-to-br from-blue-50 to-white shadow-lg">
      <h3 className="text-center mb-4 text-blue-900 text-lg">Verification Result</h3>

      <div className="space-y-4">
        {/* Similarity Score */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-blue-700">Similarity Score</span>
            <span className="font-mono text-lg text-blue-900">{(similarity * 100).toFixed(2)}%</span>
          </div>
          <Progress value={similarity * 100} className="h-3" />
        </div>

        {/* Decision */}
        <div>
          <div className="text-sm text-blue-700 mb-2">Decision</div>
          <div className="flex items-center justify-center gap-2 p-3 bg-white rounded-lg border-2" 
               style={{ borderColor: isSamePerson ? '#10b981' : '#ef4444' }}>
            {isSamePerson ? (
              <>
                <CheckCircle2 className="w-6 h-6 text-green-500" />
                <Badge className="bg-green-500 hover:bg-green-600 text-white px-4 py-1">
                  Same Person
                </Badge>
              </>
            ) : (
              <>
                <XCircle className="w-6 h-6 text-red-500" />
                <Badge className="bg-red-500 hover:bg-red-600 text-white px-4 py-1">
                  Different Person
                </Badge>
              </>
            )}
          </div>
        </div>

        {/* Threshold */}
        <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
          <span className="text-sm text-blue-700">Decision Threshold</span>
          <span className="font-mono text-blue-900">{threshold}</span>
        </div>

        {/* Inference Time */}
        <div className="flex justify-between items-center p-3 bg-blue-100 rounded-lg">
          <span className="text-sm text-blue-700">Inference Time</span>
          <span className="font-mono text-blue-900">{inferenceTime} ms</span>
        </div>
      </div>
    </Card>
  );
}
