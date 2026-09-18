const JSON_HEADERS = { 'Content-Type': 'application/json' }

/** FastAPI reports failures as {"detail": "..."}; surface that rather than a
 *  bare status line, so a user sees "Only .pdf files are accepted." */
async function errorFrom(res) {
  try {
    const body = await res.json()
    if (body?.detail) return new Error(body.detail)
  } catch {
    // no JSON body (proxy error, network failure) -- fall through to the status
  }
  return new Error(`${res.status} ${res.statusText}`)
}

async function getJson(path) {
  const res = await fetch(path)
  if (!res.ok) throw await errorFrom(res)
  return res.json()
}

export const fetchHealth = () => getJson('/api/health')
export const fetchDocuments = () => getJson('/api/documents')

/**
 * POST a PDF to /api/documents as multipart form data.
 *
 * No Content-Type header is set on purpose: the browser must write it itself
 * so it carries the multipart boundary. Setting it by hand silently breaks the
 * upload. The request resolves only once the server has finished ingesting, so
 * expect tens of seconds for a large PDF.
 */
export async function uploadDocument(file, { signal } = {}) {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch('/api/documents', { method: 'POST', body: form, signal })
  if (!res.ok) throw await errorFrom(res)
  return res.json()
}

/**
 * DELETE /api/documents/<name>.
 *
 * The name is encoded because it can contain spaces and other characters that
 * are not path-safe. The server only accepts uploaded files here; deleting a
 * document from the curated corpus answers 403.
 */
export async function deleteDocument(source) {
  const res = await fetch(`/api/documents/${encodeURIComponent(source)}`, { method: 'DELETE' })
  if (!res.ok) throw await errorFrom(res)
  return res.json()
}

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
