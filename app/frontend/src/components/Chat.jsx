import { useEffect, useRef, useState } from 'react'
import { streamChat } from '../api'
import Sources from './Sources'

const SUGGESTIONS = [
  'What is the subject matter of the NIS2 directive?',
  'Which cybersecurity risk-management measures are required?',
  'Who counts as an essential entity?',
]

export default function Chat({ hasDocuments }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  async function send(question) {
    const text = question.trim()
    if (!text || busy) return

    // History excludes the message being sent, and drops sources — the model
    // only needs the conversation, not the citation payload.
    const history = messages.map(({ role, content }) => ({ role, content }))

    setMessages((m) => [...m, { role: 'user', content: text }])
    setInput('')
    setError(null)
    setBusy(true)

    const controller = new AbortController()
    abortRef.current = controller
    let started = false

    try {
      await streamChat({
        question: text,
        history,
        signal: controller.signal,
        onSources: (sources) => {
          setMessages((m) => [...m, { role: 'assistant', content: '', sources }])
          started = true
        },
        onToken: (token) => {
          setMessages((m) => {
            const next = [...m]
            const last = next[next.length - 1]
            next[next.length - 1] = { ...last, content: last.content + token }
            return next
          })
        },
      })
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message)
        if (started) setMessages((m) => m.slice(0, -1))
      }
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send(input)
    }
  }

  return (
    <main className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="welcome">
            <h2>Ask about the directive</h2>
            {!hasDocuments && (
              <p className="warn">
                Nothing is ingested yet, so answers will have no context to draw on.
              </p>
            )}
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => send(s)} disabled={busy}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <article key={i} className={`msg ${m.role}`}>
            <div className="who">{m.role === 'user' ? 'You' : 'Assistant'}</div>
            <div className="bubble">
              {m.content || <span className="typing">▍</span>}
              {m.role === 'assistant' && <Sources sources={m.sources} />}
            </div>
          </article>
        ))}

        {error && <p className="error">{error}</p>}
        <div ref={bottomRef} />
      </div>

      <div className="composer">
        <textarea
          rows={1}
          value={input}
          placeholder="Ask a question about NIS2…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={busy}
        />
        {busy ? (
          <button className="stop" onClick={() => abortRef.current?.abort()}>
            Stop
          </button>
        ) : (
          <button className="send" onClick={() => send(input)} disabled={!input.trim()}>
            Send
          </button>
        )}
      </div>
    </main>
  )
}
