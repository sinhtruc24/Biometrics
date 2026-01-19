import React, { useState } from "react";
import { UploadCard } from "./components/UploadCard";
import { ResultPanel } from "./components/ResultPanel";
import { Button } from "./components/ui/button";
import { Alert, AlertDescription } from "./components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Loader2, AlertCircle, UserPlus, Search, Users, Database, CheckCircle, XCircle } from "lucide-react";

export default function App() {
  // Face Verification states
  const [imageA, setImageA] = useState<string | null>(null);
  const [imageB, setImageB] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<{
    similarity: number;
    isSamePerson: boolean;
    inferenceTime: number;
    threshold: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Face Registration states
  const [registerImage, setRegisterImage] = useState<string | null>(null);
  const [personName, setPersonName] = useState("");
  const [personDescription, setPersonDescription] = useState("");
  const [isRegistering, setIsRegistering] = useState(false);
  const [registerResult, setRegisterResult] = useState<any>(null);
  const [registerError, setRegisterError] = useState<string | null>(null);

  // Face Recognition states
  const [recognizeImage, setRecognizeImage] = useState<string | null>(null);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [recognizeResult, setRecognizeResult] = useState<any>(null);
  const [recognizeError, setRecognizeError] = useState<string | null>(null);
  const [topK, setTopK] = useState(5);
  const [recognitionThreshold, setRecognitionThreshold] = useState(0.25);

  // Database Management states
  const [persons, setPersons] = useState<any[]>([]);
  const [isLoadingPersons, setIsLoadingPersons] = useState(false);
  const [databaseStats, setDatabaseStats] = useState<any>(null);
  const [isLoadingStats, setIsLoadingStats] = useState(false);

  // API base URL
  const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

  // Helper function to convert base64 to File
  const base64ToFile = async (base64: string, filename: string): Promise<File> => {
    const response = await fetch(base64);
    const blob = await response.blob();
    return new File([blob], filename, { type: 'image/jpeg' });
  };

  // Face Verification functions
  const handleImageAChange = (file: File | null) => {
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImageA(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setImageA(null);
    }
  };

  const handleImageBChange = (file: File | null) => {
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImageB(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setImageB(null);
    }
  };

  const handleVerify = async () => {
    if (!imageA || !imageB) return;

    setIsProcessing(true);
    setResult(null);
    setError(null);

    try {
      const imageAFile = await base64ToFile(imageA, 'imageA.jpg');
      const imageBFile = await base64ToFile(imageB, 'imageB.jpg');

      const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
      if (imageAFile.size > MAX_FILE_SIZE || imageBFile.size > MAX_FILE_SIZE) {
        throw new Error('File size too large. Maximum 10MB per image.');
      }

      const formData = new FormData();
      formData.append('file_a', imageAFile);
      formData.append('file_b', imageBFile);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000);

      const response = await fetch(`${API_BASE_URL}/verify_faces`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      setResult({
        similarity: data.similarity,
        isSamePerson: data.is_same_person,
        inferenceTime: data.inference_time,
        threshold: data.threshold,
      });
    } catch (error) {
      console.error('Verification failed:', error);
      let errorMessage = 'Unknown error occurred';

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Request timeout. Model inference took too long. Please try again.';
        } else if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
          errorMessage = 'Network error. Please check if backend is running and CORS is configured correctly.';
        } else {
          errorMessage = error.message;
        }
      }

      setError(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  // Face Registration functions
  const handleRegisterImageChange = (file: File | null) => {
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setRegisterImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setRegisterImage(null);
    }
  };

  const handleRegister = async () => {
    if (!registerImage || !personName.trim()) return;

    setIsRegistering(true);
    setRegisterResult(null);
    setRegisterError(null);

    try {
      const imageFile = await base64ToFile(registerImage, 'register.jpg');

      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('name', personName.trim());
      formData.append('description', personDescription.trim() || "");
      formData.append('age', '25'); // Có thể thêm input để người dùng nhập tuổi
      formData.append('additional_info', JSON.stringify({})); // Thêm info bổ sung nếu cần

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000);

      const response = await fetch(`${API_BASE_URL}/register_face`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setRegisterResult(data);

      // Clear form on success
      if (data.success) {
        setRegisterImage(null);
        setPersonName("");
        setPersonDescription("");
        // Refresh persons list
        loadPersons();
      }
    } catch (error) {
      console.error('Registration failed:', error);
      let errorMessage = 'Unknown error occurred';

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Request timeout. Registration took too long. Please try again.';
        } else if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
          errorMessage = 'Network error. Please check if backend is running.';
        } else {
          errorMessage = error.message;
        }
      }

      setRegisterError(errorMessage);
    } finally {
      setIsRegistering(false);
    }
  };

  // Face Recognition functions
  const handleRecognizeImageChange = (file: File | null) => {
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setRecognizeImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setRecognizeImage(null);
    }
  };

  const handleRecognize = async () => {
    if (!recognizeImage) return;

    setIsRecognizing(true);
    setRecognizeResult(null);
    setRecognizeError(null);

    try {
      const imageFile = await base64ToFile(recognizeImage, 'recognize.jpg');

      const formData = new FormData();
      formData.append('file', imageFile);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000);

      const response = await fetch(`${API_BASE_URL}/recognize_face?top_k=${topK}&threshold=${recognitionThreshold}`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setRecognizeResult(data);
    } catch (error) {
      console.error('Recognition failed:', error);
      let errorMessage = 'Unknown error occurred';

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Request timeout. Recognition took too long. Please try again.';
        } else if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
          errorMessage = 'Network error. Please check if backend is running.';
        } else {
          errorMessage = error.message;
        }
      }

      setRecognizeError(errorMessage);
    } finally {
      setIsRecognizing(false);
    }
  };

  // Database Management functions
  const loadPersons = async () => {
    setIsLoadingPersons(true);
    try {
      const response = await fetch(`${API_BASE_URL}/persons?limit=100`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setPersons(data.persons || []);
    } catch (error) {
      console.error('Failed to load persons:', error);
    } finally {
      setIsLoadingPersons(false);
    }
  };

  const loadDatabaseStats = async () => {
    setIsLoadingStats(true);
    try {
      const response = await fetch(`${API_BASE_URL}/database/stats`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setDatabaseStats(data.stats);
    } catch (error) {
      console.error('Failed to load database stats:', error);
    } finally {
      setIsLoadingStats(false);
    }
  };

  const handleRemovePerson = async (personId: number) => {
    try {
      const response = await fetch(`${API_BASE_URL}/persons/${personId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.success) {
        // Refresh persons list
        loadPersons();
        loadDatabaseStats();
      }
    } catch (error) {
      console.error('Failed to remove person:', error);
      alert('Failed to remove person: ' + (error as Error).message);
    }
  };

  // Load data on component mount
  React.useEffect(() => {
    loadPersons();
    loadDatabaseStats();
  }, []);

  const canVerify = imageA && imageB && !isProcessing;
  const canRegister = registerImage && personName.trim() && !isRegistering;
  const canRecognize = recognizeImage && !isRecognizing;

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-blue-100">
      {/* Header */}
      <header className="bg-white border-b border-blue-200 py-4">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-xl font-bold text-blue-900 mb-1">
            Face Recognition & Verification System
          </h1>
          <p className="text-sm text-blue-700">
            GhostFaceNet + AdaFace + FAISS Vector Database
          </p>
          <p className="text-xs text-blue-600 mt-1">
            Input: 112×112 pixels • Embedding: 512 dimensions • Database: FAISS IndexFlatIP
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6 max-w-6xl">
        <Tabs defaultValue="verify" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="verify" className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4" />
              Verify
            </TabsTrigger>
            <TabsTrigger value="register" className="flex items-center gap-2">
              <UserPlus className="w-4 h-4" />
              Register
            </TabsTrigger>
            <TabsTrigger value="recognize" className="flex items-center gap-2">
              <Search className="w-4 h-4" />
              Recognize
            </TabsTrigger>
            <TabsTrigger value="database" className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              Database
            </TabsTrigger>
          </TabsList>

          {/* Face Verification Tab */}
          <TabsContent value="verify" className="space-y-6">
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <UploadCard
                  label="Upload Face Image A"
                  image={imageA}
                  onImageChange={handleImageAChange}
                />
                <UploadCard
                  label="Upload Face Image B"
                  image={imageB}
                  onImageChange={handleImageBChange}
                />
              </div>

              {error && (
                <div className="max-w-md mx-auto">
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                </div>
              )}

              <div className="text-center">
                <Button
                  onClick={handleVerify}
                  disabled={!canVerify}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-4 text-base disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Running inference...
                    </>
                  ) : (
                    "Verify Identity"
                  )}
                </Button>
              </div>

              {result && (
                <div className="max-w-2xl mx-auto">
                  <ResultPanel
                    similarity={result.similarity}
                    isSamePerson={result.isSamePerson}
                    inferenceTime={result.inferenceTime}
                    threshold={result.threshold}
                    isVisible={true}
                  />
                </div>
              )}
            </div>
          </TabsContent>

          {/* Face Registration Tab */}
          <TabsContent value="register" className="space-y-6">
            <div className="max-w-2xl mx-auto space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Register New Person</CardTitle>
                  <CardDescription>
                    Upload a face image and add person information to the database
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-center">
                    <div className="w-full max-w-sm">
                      <UploadCard
                        label="Upload Face Image"
                        image={registerImage}
                        onImageChange={handleRegisterImageChange}
                      />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="personName">Full Name *</Label>
                      <Input
                        id="personName"
                        value={personName}
                        onChange={(e) => setPersonName(e.target.value)}
                        placeholder="Enter person's full name"
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label htmlFor="personDescription">Description (Optional)</Label>
                      <Input
                        id="personDescription"
                        value={personDescription}
                        onChange={(e) => setPersonDescription(e.target.value)}
                        placeholder="Additional information about the person"
                        className="mt-1"
                      />
                    </div>
                  </div>

                  {registerError && (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>{registerError}</AlertDescription>
                    </Alert>
                  )}

                  {registerResult && registerResult.success && (
                    <Alert>
                      <CheckCircle className="h-4 w-4" />
                      <AlertDescription>
                        Successfully registered {registerResult.name} (ID: {registerResult.person_id})
                      </AlertDescription>
                    </Alert>
                  )}

                  <Button
                    onClick={handleRegister}
                    disabled={!canRegister}
                    className="w-full bg-green-600 hover:bg-green-700"
                  >
                    {isRegistering ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Registering...
                      </>
                    ) : (
                      <>
                        <UserPlus className="w-4 h-4 mr-2" />
                        Register Person
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Face Recognition Tab */}
          <TabsContent value="recognize" className="space-y-6">
            <div className="max-w-2xl mx-auto space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Recognize Face</CardTitle>
                  <CardDescription>
                    Upload a face image to search for matches in the database
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-center">
                    <div className="w-full max-w-sm">
                      <UploadCard
                        label="Upload Face Image"
                        image={recognizeImage}
                        onImageChange={handleRecognizeImageChange}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="topK">Top K Results</Label>
                      <Input
                        id="topK"
                        type="number"
                        value={topK}
                        onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
                        min="1"
                        max="20"
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label htmlFor="threshold">Threshold</Label>
                      <Input
                        id="threshold"
                        type="number"
                        value={recognitionThreshold}
                        onChange={(e) => setRecognitionThreshold(parseFloat(e.target.value) || 0.25)}
                        min="0"
                        max="1"
                        step="0.01"
                        className="mt-1"
                      />
                    </div>
                  </div>

                  {recognizeError && (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>{recognizeError}</AlertDescription>
                    </Alert>
                  )}

                  <Button
                    onClick={handleRecognize}
                    disabled={!canRecognize}
                    className="w-full bg-purple-600 hover:bg-purple-700"
                  >
                    {isRecognizing ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Recognizing...
                      </>
                    ) : (
                      <>
                        <Search className="w-4 h-4 mr-2" />
                        Recognize Face
                      </>
                    )}
                  </Button>

                  {recognizeResult && (
                    <div className="space-y-4">
                      <div className="flex items-center gap-2">
                        {recognizeResult.recognized ? (
                          <Badge variant="default" className="bg-green-100 text-green-800">
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Recognized
                          </Badge>
                        ) : (
                          <Badge variant="secondary">
                            <XCircle className="w-3 h-3 mr-1" />
                            Not Recognized
                          </Badge>
                        )}
                        <span className="text-sm text-gray-600">
                          {recognizeResult.message}
                        </span>
                      </div>

                      {recognizeResult.matches && recognizeResult.matches.length > 0 && (
                        <div className="space-y-2">
                          <h4 className="font-medium">Matches:</h4>
                          {recognizeResult.matches.map((match: any, index: number) => (
                            <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                              <div>
                                <span className="font-medium">{match.name}</span>
                                <span className="text-sm text-gray-600 ml-2">ID: {match.person_id}</span>
                              </div>
                              <Badge variant="outline">
                                {(match.similarity * 100).toFixed(1)}% similarity
                              </Badge>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Database Management Tab */}
          <TabsContent value="database" className="space-y-6">
            <div className="space-y-6">
              {/* Database Stats */}
              <Card>
                <CardHeader>
                  <CardTitle>Database Statistics</CardTitle>
                  <CardDescription>Overview of the face recognition database</CardDescription>
                </CardHeader>
                <CardContent>
                  {isLoadingStats ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="w-6 h-6 animate-spin" />
                    </div>
                  ) : databaseStats ? (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600">{databaseStats.total_persons || 0}</div>
                        <div className="text-sm text-gray-600">Total Persons</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">{databaseStats.dimension || 512}</div>
                        <div className="text-sm text-gray-600">Embedding Dim</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-600">FAISS</div>
                        <div className="text-sm text-gray-600">IndexFlatIP</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-orange-600">Cosine</div>
                        <div className="text-sm text-gray-600">Similarity</div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      Failed to load database statistics
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Registered Persons */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Users className="w-5 h-5" />
                    Registered Persons
                  </CardTitle>
                  <CardDescription>
                    List of all persons registered in the database
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {isLoadingPersons ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="w-6 h-6 animate-spin" />
                    </div>
                  ) : persons.length > 0 ? (
                    <div className="space-y-3">
                      {persons.map((person) => (
                        <div key={person.id} className="flex items-center justify-between p-4 border rounded-lg">
                          <div>
                            <div className="font-medium">{person.name}</div>
                            <div className="text-sm text-gray-600">
                              ID: {person.id} • Registered: {new Date(person.registered_at).toLocaleDateString()}
                            </div>
                            {person.description && (
                              <div className="text-sm text-gray-500 mt-1">{person.description}</div>
                            )}
                          </div>
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => handleRemovePerson(person.id)}
                          >
                            Remove
                          </Button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      No persons registered yet. Go to Register tab to add faces.
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-blue-200 py-4 mt-8">
        <div className="container mx-auto px-4 text-center">
          <div className="space-y-1">
            <p className="text-xs text-blue-600">
              Face Recognition & Verification System - Academic & Research Demo
            </p>
            <p className="text-xs text-blue-500">
              Backend: FastAPI + TensorFlow + FAISS • Frontend: React + TypeScript • Database: Persistent Vector Store
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
