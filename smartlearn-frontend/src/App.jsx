import { useState } from "react";
import { uploadPDF, askQuestion } from "./api.js";
import PdfUploader from "./PdfUploader.jsx";
import ChatPanel from "./ChatPanel.jsx";

export default function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | uploading | asking
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    if (!file) return;
    setError(null);
    setStatus("uploading");
    try {
      const data = await uploadPDF(file);
      setUpload(data);
      setFile(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setStatus("idle");
    }
  };

  const handleAsk = async () => {
    const q = message.trim();
    if (!q) return;
    setError(null);
    setStatus("asking");
    try {
      const data = await askQuestion(q);
      setAnswer(data);
      setMessage("");
    } catch (err) {
      setError(err.message);
    } finally {
      setStatus("idle");
    }
  };

  return (
    <main>
      <div className="header">
        <h1>SmartLearn AI</h1>
        <p>Upload a PDF and ask questions about your course material.</p>
      </div>

      {/* ---- upload ---- */}
      <PdfUploader
        file={file}
        onFileChange={setFile}
        upload={upload}
        status={status}
        onUpload={handleUpload}
      />

      {/* ---- error ---- */}
      {error && <p className="error" role="alert">{error}</p>}

      <hr />

      {/* ---- chat ---- */}
      <ChatPanel
        message={message}
        onMessageChange={setMessage}
        answer={answer}
        hasUpload={!!upload}
        status={status}
        onAsk={handleAsk}
      />
    </main>
  );
}
