export default function ChatPanel({ message, onMessageChange, answer, hasUpload, status, onAsk }) {
  const busy = status !== "idle";

  return (
    <>
      <section className="card">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            onAsk();
          }}
        >
          <label htmlFor="chat-input">Your question</label>
          <input
            id="chat-input"
            type="text"
            placeholder="Ask a question about the PDF…"
            value={message}
            disabled={!hasUpload || busy}
            onChange={(e) => onMessageChange(e.target.value)}
          />
          <button type="submit" disabled={!message || !hasUpload || busy}>
            Ask
          </button>
        </form>
      </section>

      {status === "asking" && <p className="status-text">Thinking…</p>}

      {answer && (
        <section className="answer-section">
          <h2>Answer</h2>
          <p>{answer.answer}</p>
          {answer.citations.length > 0 && (
            <div className="citations">
              {answer.citations.map((page) => (
                <span key={page} className="chip">
                  Page {page}
                </span>
              ))}
            </div>
          )}
        </section>
      )}
    </>
  );
}
