import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import time
from pathlib import Path
import torch

class NumpyDeprecationScraper:
    def __init__(self):
        self.base_url = "https://numpy.org/doc/stable/release/"
        self.versions = [
            "2.3.0", "2.2.6", "2.2.5", "2.2.4", "2.2.3", "2.2.2", "2.2.1", "2.2.0",
            "2.1.3", "2.1.2", "2.1.1", "2.1.0", "2.0.2", "2.0.1", "2.0.0",
            "1.26.4", "1.26.3", "1.26.2", "1.26.1", "1.26.0",
            "1.25.2", "1.25.1", "1.25.0",
            "1.24.4", "1.24.3", "1.24.2", "1.24.1", "1.24.0",
            "1.23.5", "1.23.4", "1.23.3", "1.23.2", "1.23.1", "1.23.0",
            "1.22.4", "1.22.3", "1.22.2", "1.22.1", "1.22.0",
            "1.21.6", "1.21.5", "1.21.4", "1.21.3", "1.21.2", "1.21.1", "1.21.0",
            "1.20.3", "1.20.2", "1.20.1", "1.20.0",
            "1.19.5", "1.19.4", "1.19.3", "1.19.2", "1.19.1", "1.19.0",
            "1.18.5", "1.18.4", "1.18.3", "1.18.2", "1.18.1", "1.18.0",
            "1.17.5", "1.17.4", "1.17.3", "1.17.2", "1.17.1", "1.17.0",
            "1.16.6", "1.16.5", "1.16.4", "1.16.3", "1.16.2", "1.16.1", "1.16.0"
        ]
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Embedding device: {device}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "Header 2"), ("###", "Header 3")],
            strip_headers=False
        )
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        db_path = Path("data/docs/chroma_db")
        db_path.mkdir(parents=True, exist_ok=True)
        self.chroma = Chroma(
            collection_name="numpy_docs",
            persist_directory=str(db_path),
            embedding_function=self.embeddings
        )

    def fetch_release_notes(self, version: str):
        url = f"{self.base_url}{version}-notes.html"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return BeautifulSoup(r.content, "html.parser")

    def extract_relevant_md(self, soup):
        header = soup.find(lambda tag: tag.name in ["h2", "h3"] and "deprecat" in tag.text.lower())
        if not header:
            return None
        parts = [f"{header.name} {header.get_text(strip=True)}"]
        for sib in header.find_next_siblings():
            if sib.name in ["h2", "h3"]:
                break
            text = sib.get_text("\n", strip=True)
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def chunk_and_ingest(self, markdown, version, url):
        header_docs = self.header_splitter.split_text(markdown)
        all_docs = []
        for doc in header_docs:
            subchunks = self.char_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(subchunks):
                if "deprecated" not in chunk.lower():
                    continue
                metadata = {
                    "package": "numpy",
                    "version": version,
                    "url": url,
                    "chunk_idx": i,
                    "has_deprecation": True
                }
                all_docs.append(Document(page_content=chunk, metadata=metadata))
        if all_docs:
            self.chroma.add_documents(all_docs)
            print(f"Stored {len(all_docs)} chunks for {version}")

    def run(self):
        for v in self.versions:
            print(f"\nProcessing NumPy {v}")
            soup = self.fetch_release_notes(v)
            md = self.extract_relevant_md(soup)
            if md:
                self.chunk_and_ingest(md, v, f"{self.base_url}{v}-notes.html")
            else:
                print("  → No Deprecations section found")
            time.sleep(0.5)
        print("\n✅ Done.")

if __name__ == "__main__":
    NumpyDeprecationScraper().run()
