export default function PdfUploader({ file, onFileChange, upload, status, onUpload }) {
  const busy = status !== "idle";

  return (
    <section className="card">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onUpload();
        }}
      >
        <label htmlFor="pdf-file">PDF file</label>
        <input
          id="pdf-file"
          type="file"
          accept=".pdf"
          disabled={busy}
          onChange={(e) => onFileChange(e.target.files[0] ?? null)}
        />
        {file && <p className="selected-file">Selected: {file.name}</p>}
        <button type="submit" disabled={!file || busy}>
          Upload
        </button>
      </form>

      {status === "uploading" && <p className="status-text">Uploading…</p>}

      {upload && (
        <p className="upload-result">
          ✅ Uploaded <strong>{upload.filename}</strong> — {upload.pages} page(s),{" "}
          {upload.characters} character(s)
        </p>
      )}
    </section>
  );
}
