import { useState, useRef, useCallback } from "react";

const API_BASE = "http://localhost:8000/api";

const palette = {
  saffron: "#FF9933",
  navy: "#000080",
  green: "#138808",
  accent: "#6C3FC8",
  accentLight: "#EDE9FB",
  surface: "#FAFAF8",
  surfaceAlt: "#F3F1EC",
  border: "rgba(0,0,0,0.1)",
  text: "#1A1A2E",
  muted: "#6B6B7E",
  success: "#138808",
  warning: "#FF9933",
};

const fonts = `
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');
`;

const globalStyle = `
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'DM Sans', sans-serif; background: ${palette.surface}; color: ${palette.text}; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(108,63,200,0.25); border-radius: 10px; }

  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  @keyframes shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
  }
  @keyframes barGrow { from { width: 0; } to { width: var(--target-width); } }

  .fade-in { animation: fadeIn 0.45s ease forwards; }
  .shimmer {
    background: linear-gradient(90deg, ${palette.surfaceAlt} 25%, #E8E5DE 50%, ${palette.surfaceAlt} 75%);
    background-size: 400px 100%;
    animation: shimmer 1.4s infinite;
    border-radius: 6px;
  }
`;

/* ── Sub-components ──────────────────────────────────────────────────── */

function Spinner() {
  return (
    <div style={{
      width: 28, height: 28,
      border: `3px solid ${palette.accentLight}`,
      borderTopColor: palette.accent,
      borderRadius: "50%",
      animation: "spin 0.75s linear infinite",
      display: "inline-block",
    }} />
  );
}

function Badge({ children, color = palette.accent, bg = palette.accentLight }) {
  return (
    <span style={{
      background: bg, color, fontSize: 11, fontWeight: 500,
      padding: "3px 9px", borderRadius: 20, letterSpacing: "0.03em",
      textTransform: "uppercase", whiteSpace: "nowrap",
    }}>
      {children}
    </span>
  );
}

function StatCard({ label, value, unit = "", accent = palette.accent }) {
  return (
    <div style={{
      background: "#FFF", border: `0.5px solid ${palette.border}`,
      borderRadius: 12, padding: "14px 18px",
      borderLeft: `3px solid ${accent}`,
    }}>
      <p style={{ fontSize: 11, color: palette.muted, marginBottom: 4, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.06em" }}>
        {label}
      </p>
      <p style={{ fontSize: 22, fontWeight: 600, color: palette.text, fontFamily: "'Playfair Display', serif" }}>
        {value}<span style={{ fontSize: 13, fontWeight: 400, color: palette.muted, marginLeft: 4 }}>{unit}</span>
      </p>
    </div>
  );
}

function CompressionBar({ label, value, max, color }) {
  const pct = Math.round((value / Math.max(max, 1)) * 100);
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: palette.muted, marginBottom: 4 }}>
        <span>{label}</span>
        <span style={{ fontWeight: 500, color: palette.text }}>{value.toLocaleString()}</span>
      </div>
      <div style={{ height: 7, background: palette.surfaceAlt, borderRadius: 4, overflow: "hidden" }}>
        <div style={{
          height: "100%", borderRadius: 4, background: color,
          "--target-width": `${pct}%`, width: `${pct}%`,
          animation: "barGrow 0.8s ease forwards",
        }} />
      </div>
    </div>
  );
}

function KeyPointCard({ point, index }) {
  return (
    <div className="fade-in" style={{
      display: "flex", gap: 14, padding: "14px 18px",
      background: "#FFF", border: `0.5px solid ${palette.border}`,
      borderRadius: 10, animationDelay: `${index * 0.07}s`, opacity: 0,
    }}>
      <div style={{
        minWidth: 26, height: 26, borderRadius: "50%",
        background: palette.accentLight, color: palette.accent,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 12, fontWeight: 600,
      }}>
        {index + 1}
      </div>
      <p style={{ fontSize: 14, lineHeight: 1.65, color: palette.text }}>{point}</p>
    </div>
  );
}

function UploadZone({ onFile, isLoading }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onFile(file);
  }, [onFile]);

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onFile(file);
  };

  return (
    <div
      onClick={() => !isLoading && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      style={{
        border: `2px dashed ${dragging ? palette.accent : palette.border}`,
        borderRadius: 14, padding: "36px 24px", textAlign: "center",
        cursor: isLoading ? "not-allowed" : "pointer",
        background: dragging ? palette.accentLight : "#FFF",
        transition: "all 0.2s",
      }}
    >
      <input ref={inputRef} type="file" accept=".pdf,.txt" onChange={handleChange} style={{ display: "none" }} />
      <div style={{ fontSize: 36, marginBottom: 10 }}>
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none" style={{ margin: "0 auto", display: "block" }}>
          <rect width="40" height="40" rx="10" fill={palette.accentLight} />
          <path d="M20 26V14M20 14L15 19M20 14L25 19" stroke={palette.accent} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M12 28h16" stroke={palette.accent} strokeWidth="2" strokeLinecap="round"/>
        </svg>
      </div>
      {isLoading ? (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
          <Spinner />
          <span style={{ color: palette.muted, fontSize: 14 }}>Analysing document…</span>
        </div>
      ) : (
        <>
          <p style={{ fontWeight: 500, fontSize: 15, color: palette.text, marginBottom: 4 }}>
            Drop a PDF or text file here
          </p>
          <p style={{ fontSize: 13, color: palette.muted }}>
            or click to browse · supports .pdf and .txt · up to 50 MB
          </p>
        </>
      )}
    </div>
  );
}

function TextTab({ onSubmit, isLoading }) {
  const [text, setText] = useState("");
  return (
    <div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste the full text of a legislative bill, government circular, or legal document here…"
        rows={8}
        style={{
          width: "100%", resize: "vertical", padding: "14px 16px",
          border: `1px solid ${palette.border}`, borderRadius: 10,
          fontFamily: "'DM Sans', sans-serif", fontSize: 14, lineHeight: 1.7,
          background: "#FFF", color: palette.text, outline: "none",
          transition: "border-color 0.2s",
        }}
        onFocus={(e) => e.target.style.borderColor = palette.accent}
        onBlur={(e) => e.target.style.borderColor = palette.border}
      />
      <button
        disabled={!text.trim() || isLoading}
        onClick={() => onSubmit(text)}
        style={{
          marginTop: 12, padding: "11px 28px",
          background: text.trim() && !isLoading ? palette.accent : palette.surfaceAlt,
          color: text.trim() && !isLoading ? "#FFF" : palette.muted,
          border: "none", borderRadius: 8, cursor: text.trim() && !isLoading ? "pointer" : "not-allowed",
          fontFamily: "'DM Sans', sans-serif", fontSize: 14, fontWeight: 500,
          transition: "background 0.2s",
          display: "flex", alignItems: "center", gap: 8,
        }}
      >
        {isLoading ? <><Spinner /><span>Analysing…</span></> : "Analyse Document →"}
      </button>
    </div>
  );
}

/* ── Result Panel ─────────────────────────────────────────────────────── */
function ResultPanel({ result }) {
  const { summary, key_points = [], stats = {}, word_count } = result;
  const {
    original_tokens, compressed_tokens,
    token_reduction_pct, char_reduction_pct,
    information_density_score,
  } = stats;

  return (
    <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Stats row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 12 }}>
        {word_count && <StatCard label="Original words" value={word_count.toLocaleString()} accent={palette.navy} />}
        {original_tokens && <StatCard label="Orig. tokens" value={original_tokens.toLocaleString()} accent={palette.navy} />}
        {compressed_tokens && <StatCard label="Compressed tokens" value={compressed_tokens.toLocaleString()} accent={palette.accent} />}
        {information_density_score !== undefined && (
          <StatCard label="Info density" value={(information_density_score * 100).toFixed(1)} unit="%" accent={palette.green} />
        )}
      </div>

      {/* Compression bars */}
      {(original_tokens || original_tokens === 0) && (
        <div style={{
          background: "#FFF", border: `0.5px solid ${palette.border}`,
          borderRadius: 12, padding: "18px 20px",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <p style={{ fontWeight: 500, fontSize: 14, color: palette.text }}>Token compression</p>
            <Badge color={palette.green} bg="#E6F7EA">
              {token_reduction_pct}% saved
            </Badge>
          </div>
          <CompressionBar label="Original tokens" value={original_tokens} max={original_tokens} color={`${palette.navy}55`} />
          <CompressionBar label="Compressed tokens" value={compressed_tokens} max={original_tokens} color={palette.accent} />
          {char_reduction_pct && (
            <p style={{ fontSize: 12, color: palette.muted, marginTop: 8 }}>
              Character reduction: {char_reduction_pct}%
            </p>
          )}
        </div>
      )}

      {/* Summary */}
      <div style={{
        background: "#FFF", border: `0.5px solid ${palette.border}`,
        borderRadius: 12, padding: "20px 22px",
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <h3 style={{ fontFamily: "'Playfair Display', serif", fontWeight: 600, fontSize: 18, color: palette.text }}>
            Summary
          </h3>
          <Badge>AI-generated</Badge>
        </div>
        <p style={{ fontSize: 15, lineHeight: 1.75, color: palette.text }}>{summary}</p>
      </div>

      {/* Key points */}
      {key_points.length > 0 && (
        <div>
          <h3 style={{
            fontFamily: "'Playfair Display', serif", fontWeight: 600,
            fontSize: 18, color: palette.text, marginBottom: 12,
          }}>
            Key points ({key_points.length})
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {key_points.map((pt, i) => <KeyPointCard key={i} point={pt} index={i} />)}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Main App ─────────────────────────────────────────────────────────── */
export default function App() {
  const [tab, setTab] = useState("upload"); // "upload" | "text"
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [keepRatio, setKeepRatio] = useState(0.45);
  const [docName, setDocName] = useState("");

  const handleTextSubmit = async (text) => {
    setIsLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch(`${API_BASE}/summarize/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, keep_ratio: keepRatio }),
      });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
      setDocName("Pasted text");
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFile = async (file) => {
    setIsLoading(true); setError(null); setResult(null); setDocName(file.name);
    try {
      const isPDF = file.type === "application/pdf";
      if (isPDF) {
        // Convert to base64 and send
        const base64 = await new Promise((res, rej) => {
          const r = new FileReader();
          r.onload = () => res(r.result.split(",")[1]);
          r.onerror = () => rej(new Error("Failed to read file"));
          r.readAsDataURL(file);
        });
        const apiRes = await fetch(`${API_BASE}/summarize/pdf-base64`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pdf_base64: base64, keep_ratio: keepRatio }),
        });
        if (!apiRes.ok) throw new Error(await apiRes.text());
        setResult(await apiRes.json());
      } else {
        // Plain text file
        const text = await file.text();
        await handleTextSubmit(text);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <style>{fonts}</style>
      <style>{globalStyle}</style>

      <div style={{ minHeight: "100vh", background: palette.surface }}>
        {/* ── Header ── */}
        <header style={{
          background: "#FFF", borderBottom: `0.5px solid ${palette.border}`,
          padding: "0 24px", position: "sticky", top: 0, zIndex: 100,
        }}>
          <div style={{
            maxWidth: 960, margin: "0 auto",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            height: 60,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              {/* Tricolor dot strip */}
              <div style={{ display: "flex", gap: 3 }}>
                {[palette.saffron, "#FFF", palette.green].map((c, i) => (
                  <div key={i} style={{
                    width: 5, height: 26, borderRadius: 3,
                    background: c, border: i === 1 ? `1px solid ${palette.border}` : "none",
                  }} />
                ))}
              </div>
              <div>
                <h1 style={{
                  fontFamily: "'Playfair Display', serif",
                  fontWeight: 700, fontSize: 18, color: palette.text, lineHeight: 1.1,
                }}>
                  Legislative Analyzer
                </h1>
                <p style={{ fontSize: 11, color: palette.muted, letterSpacing: "0.04em" }}>
                  Citizen's Dashboard
                </p>
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <Badge color={palette.saffron} bg="#FFF5E6">India</Badge>
              <Badge color={palette.accent} bg={palette.accentLight}>AI-Powered</Badge>
            </div>
          </div>
        </header>

        {/* ── Hero strip ── */}
        <div style={{
          background: `linear-gradient(135deg, ${palette.navy} 0%, #1A1A6E 50%, #2D0B8A 100%)`,
          padding: "36px 24px",
          textAlign: "center",
        }}>
          <p style={{
            fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 700,
            color: "#FFF", marginBottom: 8, lineHeight: 1.3,
          }}>
            Understand India's Laws,<br />In Plain Language
          </p>
          <p style={{ fontSize: 14, color: "rgba(255,255,255,0.65)", maxWidth: 520, margin: "0 auto" }}>
            Upload any parliamentary bill, government circular, or legal document.
            Our AI compresses and summarises it with maximum information density.
          </p>
        </div>

        {/* ── Main content ── */}
        <main style={{ maxWidth: 960, margin: "0 auto", padding: "28px 24px" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>

            {/* ── Left: Input ── */}
            <div>
              <div style={{
                background: "#FFF", border: `0.5px solid ${palette.border}`,
                borderRadius: 14, overflow: "hidden",
              }}>
                {/* Tab bar */}
                <div style={{
                  display: "flex", borderBottom: `0.5px solid ${palette.border}`,
                }}>
                  {[
                    { id: "upload", label: "Upload File" },
                    { id: "text", label: "Paste Text" },
                  ].map((t) => (
                    <button
                      key={t.id}
                      onClick={() => setTab(t.id)}
                      style={{
                        flex: 1, padding: "12px 0",
                        background: tab === t.id ? palette.accentLight : "transparent",
                        color: tab === t.id ? palette.accent : palette.muted,
                        border: "none", cursor: "pointer",
                        fontFamily: "'DM Sans', sans-serif",
                        fontSize: 13, fontWeight: tab === t.id ? 500 : 400,
                        borderBottom: tab === t.id ? `2px solid ${palette.accent}` : "2px solid transparent",
                        transition: "all 0.15s",
                      }}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>

                <div style={{ padding: "18px 20px" }}>
                  {tab === "upload"
                    ? <UploadZone onFile={handleFile} isLoading={isLoading} />
                    : <TextTab onSubmit={handleTextSubmit} isLoading={isLoading} />
                  }
                </div>
              </div>

              {/* ── Compression tuner ── */}
              <div style={{
                marginTop: 16, background: "#FFF",
                border: `0.5px solid ${palette.border}`,
                borderRadius: 12, padding: "16px 20px",
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                  <div>
                    <p style={{ fontWeight: 500, fontSize: 13, color: palette.text }}>Compression ratio</p>
                    <p style={{ fontSize: 11, color: palette.muted }}>Lower = more compressed, fewer tokens</p>
                  </div>
                  <span style={{
                    background: palette.accentLight, color: palette.accent,
                    fontWeight: 600, fontSize: 15, padding: "4px 12px", borderRadius: 8,
                  }}>
                    {Math.round(keepRatio * 100)}%
                  </span>
                </div>
                <input
                  type="range" min="10" max="90" step="5"
                  value={Math.round(keepRatio * 100)}
                  onChange={(e) => setKeepRatio(Number(e.target.value) / 100)}
                  style={{ width: "100%", accentColor: palette.accent }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: palette.muted, marginTop: 4 }}>
                  <span>Dense (10%)</span>
                  <span>Verbose (90%)</span>
                </div>
              </div>

              {/* How it works */}
              <div style={{
                marginTop: 16, background: "#FFF",
                border: `0.5px solid ${palette.border}`,
                borderRadius: 12, padding: "16px 20px",
              }}>
                <p style={{ fontWeight: 500, fontSize: 13, color: palette.text, marginBottom: 12 }}>How it works</p>
                {[
                  { step: "1", title: "Parse", desc: "PDF extracted & cleaned", color: palette.saffron },
                  { step: "2", title: "Compress", desc: "TF-IDF token compression", color: palette.accent },
                  { step: "3", title: "Chunk", desc: "Safe ~900-token chunks", color: palette.navy },
                  { step: "4", title: "Summarise", desc: "HuggingFace BART model", color: palette.green },
                ].map((s, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 12, marginBottom: i < 3 ? 10 : 0 }}>
                    <div style={{
                      minWidth: 26, height: 26, borderRadius: "50%",
                      background: s.color, color: "#FFF",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 12, fontWeight: 600,
                    }}>{s.step}</div>
                    <div>
                      <p style={{ fontSize: 13, fontWeight: 500, color: palette.text }}>{s.title}</p>
                      <p style={{ fontSize: 11, color: palette.muted }}>{s.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Right: Results ── */}
            <div>
              {isLoading && (
                <div style={{
                  background: "#FFF", border: `0.5px solid ${palette.border}`,
                  borderRadius: 14, padding: 32, textAlign: "center",
                }}>
                  <Spinner />
                  <p style={{ marginTop: 16, color: palette.text, fontWeight: 500 }}>Analysing document…</p>
                  <p style={{ marginTop: 6, fontSize: 13, color: palette.muted }}>
                    Compressing tokens and generating summary
                  </p>
                  {/* Skeleton */}
                  <div style={{ marginTop: 24, textAlign: "left" }}>
                    {[80, 60, 90, 70].map((w, i) => (
                      <div key={i} className="shimmer" style={{ height: 14, width: `${w}%`, marginBottom: 10 }} />
                    ))}
                  </div>
                </div>
              )}

              {error && (
                <div style={{
                  background: "#FFF9F9", border: `1px solid #FFCDD2`,
                  borderRadius: 12, padding: "18px 20px",
                }}>
                  <p style={{ fontWeight: 500, color: "#C62828", marginBottom: 6 }}>Error</p>
                  <p style={{ fontSize: 13, color: "#B71C1C", lineHeight: 1.6 }}>{error}</p>
                  <p style={{ fontSize: 12, color: palette.muted, marginTop: 8 }}>
                    Make sure the backend is running at {API_BASE}
                  </p>
                </div>
              )}

              {!isLoading && !error && !result && (
                <div style={{
                  background: "#FFF", border: `0.5px solid ${palette.border}`,
                  borderRadius: 14, padding: "48px 32px", textAlign: "center",
                }}>
                  <svg width="56" height="56" viewBox="0 0 56 56" fill="none" style={{ margin: "0 auto 16px" }}>
                    <rect width="56" height="56" rx="14" fill={palette.accentLight} />
                    <path d="M18 20h20M18 27h20M18 34h12" stroke={palette.accent} strokeWidth="2" strokeLinecap="round"/>
                    <circle cx="40" cy="38" r="8" fill={palette.accentLight} stroke={palette.accent} strokeWidth="1.5"/>
                    <path d="M37 38l2 2 4-3" stroke={palette.accent} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <p style={{ fontFamily: "'Playfair Display', serif", fontSize: 18, fontWeight: 600, color: palette.text, marginBottom: 8 }}>
                    Ready to analyse
                  </p>
                  <p style={{ fontSize: 13, color: palette.muted, lineHeight: 1.7 }}>
                    Upload a PDF or paste text on the left.<br />
                    The AI will compress and summarise it for you.
                  </p>
                </div>
              )}

              {!isLoading && !error && result && (
                <div>
                  {docName && (
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                      <Badge color={palette.navy} bg="#E8EAF6">{docName}</Badge>
                      <button
                        onClick={() => { setResult(null); setDocName(""); }}
                        style={{
                          background: "transparent", border: "none",
                          cursor: "pointer", color: palette.muted, fontSize: 12,
                        }}
                      >
                        × Clear
                      </button>
                    </div>
                  )}
                  <ResultPanel result={result} />
                </div>
              )}
            </div>
          </div>
        </main>

        {/* ── Footer ── */}
        <footer style={{
          borderTop: `0.5px solid ${palette.border}`, padding: "16px 24px",
          textAlign: "center", marginTop: 24,
        }}>
          <p style={{ fontSize: 12, color: palette.muted }}>
            AI Legislative Analyzer · Built with HuggingFace BART · Token Compression for information density
          </p>
        </footer>
      </div>
    </>
  );
}
