import { getDocumentFileURL } from "./api.js";

export default function PdfPreview({ upload, activePage, chatId }) {
  if (!upload) {
    return (
      <section className="pdf-preview">
        <div className="preview-placeholder">
          <p>Upload a PDF to preview it here.</p>
        </div>
      </section>
    );
  }

  const fileURL = getDocumentFileURL(chatId);

  return (
    <section className="pdf-preview">
      <div className="preview-toolbar">
        <span>Page {activePage}</span>
      </div>
      <iframe
        key={`${chatId}-${activePage}`}
        src={`${fileURL}#page=${activePage}`}
        title="PDF Preview"
        className="preview-frame"
      />
    </section>
  );
}
