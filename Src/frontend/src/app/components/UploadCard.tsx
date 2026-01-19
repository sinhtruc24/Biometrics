import { Upload, X } from "lucide-react";
import { Card } from "./ui/card";

interface UploadCardProps {
  label: string;
  image: string | null;
  onImageChange: (file: File | null) => void;
}

export function UploadCard({ label, image, onImageChange }: UploadCardProps) {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file type
      const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
      if (!allowedTypes.includes(file.type)) {
        alert('Please select a valid image file (JPEG, PNG, WebP)');
        return;
      }

      // Validate file size (max 10MB)
      const maxSize = 10 * 1024 * 1024; // 10MB
      if (file.size > maxSize) {
        alert('File size too large. Maximum 10MB allowed.');
        return;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        onImageChange(file);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemove = () => {
    onImageChange(null);
  };

  return (
    <Card className="relative overflow-hidden">
      <div className="p-6">
        <h3 className="mb-4 text-center text-blue-800">{label}</h3>
        
        <div className="relative aspect-square bg-blue-50 rounded-lg border-2 border-dashed border-blue-300 flex items-center justify-center overflow-hidden hover:border-blue-400 transition-colors">
          {image ? (
            <>
              <img
                src={image}
                alt={label}
                className="w-full h-full object-cover"
              />
              <button
                onClick={handleRemove}
                className="absolute top-2 right-2 bg-red-500 text-white p-1.5 rounded-full hover:bg-red-600 transition-colors"
                aria-label="Remove image"
              >
                <X className="w-4 h-4" />
              </button>
            </>
          ) : (
            <label className="cursor-pointer flex flex-col items-center justify-center w-full h-full">
              <Upload className="w-12 h-12 text-blue-400 mb-2" />
              <span className="text-sm text-blue-600">Click to upload</span>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
          )}
        </div>
      </div>
    </Card>
  );
}
