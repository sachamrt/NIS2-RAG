import { useEffect, useRef, useState } from 'react'
import { deleteDocument, fetchDocuments, fetchHealth, uploadDocument } from '../api'

export default function Sidebar({ refreshKey, onUploaded }) {
  const [health, setHealth] = useState(null)
  const [docs, setDocs] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(null)
  const [uploadError, setUploadError] = useState(null)
  const [deleting, setDeleting] = useState(null)
  const fileInput = useRef(null)

  async function load() {
    setLoading(true)
    try {
      const [h, d] = await Promise.all([fetchHealth(), fetchDocuments()])
      setHealth(h)
      setDocs(d)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [refreshKey])

  async function onDelete(source) {
    // Irreversible: the PDF is unlinked, not just un-indexed.
    if (!window.confirm(`Delete ${source} and its chunks? This cannot be undone.`)) return

    setDeleting(source)
    setUploadError(null)
    try {
      await deleteDocument(source)
      onUploaded?.()   // same refresh path as an upload
    } catch (err) {
      setUploadError(err.message)
    } finally {
      setDeleting(null)
    }
  }

  async function onPick(event) {
    const file = event.target.files?.[0]
    // Reset immediately so picking the same file twice still fires onChange.
    event.target.value = ''
    if (!file) return

    setUploading(file.name)
    setUploadError(null)
    try {
      await uploadDocument(file)
      onUploaded?.()   // App bumps refreshKey -> this sidebar and the chat reload
    } catch (err) {
      setUploadError(err.message)
    } finally {
      setUploading(null)
    }
  }

  return (
    <aside className="sidebar">
      <div className="brand">
        <h1>NIS2 RAG</h1>
        <button className="ghost" onClick={load} disabled={loading} title="Refresh">
          {loading ? '…' : '↻'}
        </button>
      </div>

      {error && <p className="error">API unreachable — {error}</p>}

      {health && (
        <dl className="status">
          <div>
            <dt>LLM</dt>
            <dd>{health.llm_model}</dd>
          </div>
          <div>
            <dt>Embeddings</dt>
            <dd>{health.embedding_provider}</dd>
          </div>
          <div>
            <dt>Collection</dt>
            <dd>{health.collection}</dd>
          </div>
          <div>
            <dt>Qdrant</dt>
            <dd className={health.qdrant_reachable ? 'ok' : 'bad'}>
              {health.qdrant_reachable ? 'connected' : 'unreachable'}
            </dd>
          </div>
        </dl>
      )}

      <h2>
        Ingested documents
        {docs?.documents.length > 0 && <span className="count">{docs.documents.length}</span>}
      </h2>

      <input
        ref={fileInput}
        type="file"
        accept="application/pdf,.pdf"
        onChange={onPick}
        hidden
      />
      <button
        className="upload"
        onClick={() => fileInput.current?.click()}
        disabled={!!uploading}
      >
        {uploading ? `Ingesting ${uploading}…` : '+ Add PDF'}
      </button>
      {uploading && (
        <p className="hint">Embedding the whole file — this can take a minute.</p>
      )}
      {uploadError && <p className="error">{uploadError}</p>}

      {docs && docs.documents.length === 0 && (
        <div className="empty">
          <p>No documents ingested yet.</p>
          <p>
            Drop PDFs into <code>data/raw_pdfs/</code> then run:
          </p>
          <code className="cmd">python scripts/ingest_cli.py</code>
        </div>
      )}

      <ul className="doclist">
        {docs?.documents.map((doc) => (
          <li key={doc.source}>
            <span className="docname" title={doc.source}>
              {doc.source}
            </span>
            <span className="chunks">{doc.chunks} chunks</span>
            {doc.deletable ? (
              <button
                className="del"
                onClick={() => onDelete(doc.source)}
                disabled={deleting === doc.source}
                title="Delete this uploaded document"
              >
                {deleting === doc.source ? '…' : '×'}
              </button>
            ) : (
              <span className="pinned" title="Curated corpus in data/raw_pdfs — not deletable here">
                ○
              </span>
            )}
          </li>
        ))}
      </ul>

      {docs?.points > 0 && <p className="total">{docs.points} chunks indexed in total</p>}
    </aside>
  )
}
