import { useEffect, useState } from 'react'
import Chat from './components/Chat'
import Sidebar from './components/Sidebar'
import { fetchDocuments } from './api'
import './App.css'

export default function App() {
  const [hasDocuments, setHasDocuments] = useState(true)
  const [refreshKey] = useState(0)

  useEffect(() => {
    fetchDocuments()
      .then((d) => setHasDocuments(d.documents.length > 0))
      .catch(() => setHasDocuments(false))
  }, [refreshKey])

  return (
    <div className="layout">
      <Sidebar refreshKey={refreshKey} />
      <Chat hasDocuments={hasDocuments} />
    </div>
  )
}
