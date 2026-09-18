import { useEffect, useState } from 'react'
import { fetchDocuments, fetchHealth } from '../api'

export default function Sidebar({ refreshKey }) {
  const [health, setHealth] = useState(null)
  const [docs, setDocs] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

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
          </li>
        ))}
      </ul>

      {docs?.points > 0 && <p className="total">{docs.points} chunks indexed in total</p>}
    </aside>
  )
}
