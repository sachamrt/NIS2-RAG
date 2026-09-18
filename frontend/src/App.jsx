import { useCallback, useEffect, useState } from 'react'
import Chat from './components/Chat'
import Sidebar from './components/Sidebar'
import { fetchDocuments } from './api'
import './App.css'

export default function App() {
  const [hasDocuments, setHasDocuments] = useState(true)
  const [refreshKey, setRefreshKey] = useState(0)

  // Bumping the key re-runs this effect and the Sidebar's own loader, so an
  // upload updates the document list and the chat's empty-state warning at once.
  const refresh = useCallback(() => setRefreshKey((k) => k + 1), [])

  useEffect(() => {
    fetchDocuments()
      .then((d) => setHasDocuments(d.documents.length > 0))
      .catch(() => setHasDocuments(false))
  }, [refreshKey])

  return (
    <div className="layout">
      <Sidebar refreshKey={refreshKey} onUploaded={refresh} />
      <Chat hasDocuments={hasDocuments} />
    </div>
  )
}
