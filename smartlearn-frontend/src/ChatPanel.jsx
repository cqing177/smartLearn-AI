import { useState } from "react";
import { askQuestion } from "./api.js";

export default function ChatPanel({ enabled, onBusy, disabled, onJumpToPage }) {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAsk = async () => {
    const q = message.trim();
    if (!q || !enabled || loading) return;

    const userMsg = { role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setMessage("");
    setError(null);
    setLoading(true);
    if (onBusy) onBusy(true);

    try {
      const data = await askQuestion(q);
      const assistantMsg = {
        role: "assistant",
        content: data.answer,
        citations: data.citations || [],
        sources: data.sources || [],
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      if (onBusy) onBusy(false);
    }
  };

  return (
    <section className="chat-panel">
      <div className="message-list">
        {messages.length === 0 && !loading && (
          <p className="chat-placeholder">
            Ask a question about the uploaded PDF.
          </p>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-role">
              {msg.role === "user" ? "You" : "Assistant"}
            </div>
            <div className="message-content">{msg.content}</div>
            {msg.citations && msg.citations.length > 0 && (
              <div className="citations">
                {msg.citations.map((page) => (
                  <button
                    key={page}
                    className="chip clickable"
                    type="button"
                    onClick={() => onJumpToPage && onJumpToPage(page)}
                  >
                    Page {page}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && <p className="status-text">Thinking…</p>}
        {error && <p className="error" role="alert">{error}</p>}
      </div>

      <form
        className="chat-input-form"
        onSubmit={(e) => {
          e.preventDefault();
          handleAsk();
        }}
      >
        <input
          type="text"
          placeholder="Ask a question about the PDF…"
          value={message}
          disabled={!enabled || loading || disabled}
          onChange={(e) => setMessage(e.target.value)}
        />
        <button type="submit" disabled={!message || !enabled || loading || disabled}>
          Ask
        </button>
      </form>
    </section>
  );
}
