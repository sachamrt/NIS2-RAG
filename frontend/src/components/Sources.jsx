import { useState } from 'react'

export default function Sources({ sources }) {
  const [open, setOpen] = useState(null)
  if (!sources?.length) return null

  return (
    <div className="sources">
      <div className="chips">
        {sources.map((s, i) => (
          <button
            key={`${s.source}-${s.page}-${s.chunk_index}`}
            className={`chip ${open === i ? 'active' : ''}`}
            onClick={() => setOpen(open === i ? null : i)}
            title={`similarity ${s.score}`}
          >
            {s.source} · p.{s.page}
          </button>
        ))}
      </div>
      {open !== null && (
        <blockquote className="excerpt">
          <span className="score">score {sources[open].score}</span>
          {sources[open].excerpt}…
        </blockquote>
      )}
    </div>
  )
}
