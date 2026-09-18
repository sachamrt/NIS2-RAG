const JSON_HEADERS = { 'Content-Type': 'application/json' }

async function getJson(path) {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const fetchHealth = () => getJson('/api/health')
export const fetchDocuments = () => getJson('/api/documents')

/**
 * POST /api/chat/stream and parse the SSE body.
 *
 * EventSource only speaks GET, and we need to POST the question plus history,
 * so the stream is read manually. Events arrive as `event: <name>\ndata: <json>`
 * separated by blank lines; a chunk can split mid-event, hence the buffer.
 */
export async function streamChat({ question, history, signal, onSources, onToken }) {
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ question, history }),
    signal,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      if (!frame.trim()) continue
      const nameLine = frame.split('\n').find((l) => l.startsWith('event: '))
      const dataLine = frame.split('\n').find((l) => l.startsWith('data: '))
      if (!nameLine || !dataLine) continue

      const name = nameLine.slice(7).trim()
      const payload = JSON.parse(dataLine.slice(6))

      if (name === 'sources') onSources(payload.sources)
      else if (name === 'token') onToken(payload.token)
      else if (name === 'error') throw new Error(payload.detail)
    }
  }
}
