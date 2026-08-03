import { useState, useCallback } from "react";
import { uploadPDF, CHAT_ID } from "./api.js";
import PdfUploader from "./PdfUploader.jsx";
import PdfPreview from "./PdfPreview.jsx";
import ChatPanel from "./ChatPanel.jsx";

export default function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | uploading
  const [error, setError] = useState(null);
  const [activePage, setActivePage] = useState(1);
  const [uploadKey, setUploadKey] = useState(0);

  const handleUpload = async () => {
    if (!file) return;
    setError(null);
    setStatus("uploading");
    try {
      const data = await uploadPDF(file);
      setUpload(data);
      setFile(null);
      setActivePage(1);
      setUploadKey((k) => k + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setStatus("idle");
    }
  };

  const handleJumpToPage = useCallback((page) => {
    setActivePage(page);
  }, []);

  return (
    <main>
      <div className="header">
        <h1>SmartLearn AI</h1>
        <p>Upload a PDF and ask questions about your course material.</p>
      </div>

      <PdfUploader
        file={file}
        onFileChange={setFile}
        upload={upload}
        status={status}
        onUpload={handleUpload}
      />

      {error && <p className="error" role="alert">{error}</p>}

      <div className="workspace">
        <PdfPreview
          upload={upload}
          activePage={activePage}
          chatId={CHAT_ID}
        />
        <ChatPanel
          key={uploadKey}
          enabled={!!upload}
          disabled={status !== "idle"}
          onJumpToPage={handleJumpToPage}
        />
      </div>
    </main>
  );
}
