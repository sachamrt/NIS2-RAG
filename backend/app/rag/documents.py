"""What has actually been ingested, read back from Qdrant.

Uses Qdrant's facet API to count chunks per source file -- server-side
aggregation, so listing stays cheap as the corpus grows.
"""

from app.core.config import get_settings
from app.core.paths import RAW_PDF_DIR
from app.core.vectorstore import get_client


def list_documents() -> dict:
    settings = get_settings()
    client = get_client()
    name = settings.qdrant_collection

    if not client.collection_exists(name):
        return {"collection": name, "points": 0, "documents": []}

    info = client.get_collection(name)
    facet = client.facet(collection_name=name, key="metadata.source", limit=100)
    documents = sorted(
        (
            {
                "source": str(hit.value),
                "chunks": hit.count,
                # Curated PDFs on disk are off-limits; uploads left no file.
                "deletable": not (RAW_PDF_DIR / str(hit.value)).exists(),
            }
            for hit in facet.hits
        ),
        key=lambda d: d["source"],
    )
    return {
        "collection": name,
        "points": info.points_count or 0,
        "documents": documents,
    }
