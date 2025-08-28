import asyncio
import hashlib
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_postgres import PGEngine
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sqlalchemy as sa
from tqdm.asyncio import tqdm_asyncio

from config import settings
from helper import setup_logging
from rag import use_pg_engine, load_embeddings, load_vector_store


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

async def main():
    setup_logging()
    load_dotenv()
    
    async with use_pg_engine() as (pg_engine, engine):
        try:
            await prepare_pg_engine(pg_engine, overwrite_existing=False)
        except sa.exc.ProgrammingError:
            pass
        await ensure_hash_index(engine)

        embeddings = load_embeddings(settings.HF_EMBEDDINGS_MODEL_NAME)
        vector_store = await load_vector_store(
            pg_engine, settings.TABLE_NAME, embeddings
        )
        
        docs = await load_data(settings.CORPUS_DIR_PATH, 'utf-8')
        splits = await split_text(
            docs, chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        for i in tqdm_asyncio(range(len(splits)), desc='Hashing'):
            splits[i].metadata['sha1_hash'] = hash_document(splits[i])

        total_chunks = 0
        for i in tqdm_asyncio(
            range(0, len(splits), settings.INDEXING_BATCH_SIZE), desc='Indexing'
        ):
            batch = splits[i:i + settings.INDEXING_BATCH_SIZE]
            batch_hashes = list(map(lambda doc: doc.metadata['sha1_hash'], batch))
            existing_hashes = await get_existing_hashes(engine, batch_hashes)
            new_chunks = list(filter(
                lambda doc: doc.metadata['sha1_hash'] not in existing_hashes,
                batch
            ))
            if new_chunks:
                await vector_store.aadd_documents(new_chunks)
                total_chunks += len(new_chunks)
                logger.info(
                    f'Added {len(new_chunks)} new chunks to the vector store.'
                )
        logger.info(f'Index build complete. Added {total_chunks} chunks into vector store')


async def prepare_pg_engine(pg_engine: PGEngine, overwrite_existing = False):
    await pg_engine.ainit_vectorstore_table(
        table_name=settings.TABLE_NAME,
        vector_size=settings.VECTOR_SIZE,
        overwrite_existing=overwrite_existing
    )


async def ensure_hash_index(engine, index_name='idx_vectorstore_metadata_sha1_hash'):
    sql = f'''
    CREATE INDEX IF NOT EXISTS {index_name}
    ON {settings.TABLE_NAME} ((langchain_metadata->>'sha1_hash'));
    '''
    async with engine.begin() as conn:
        await conn.execute(sa.text(sql))
    logger.info(f'Ensured index "{index_name}" exists.')


async def get_existing_hashes(engine, hashes: list[str]) -> set[str]:
    """
    Efficiently retrieves which of the given hashes already exist in database.
    """
    if not hashes:
        return set()
    sql = sa.text(f'''
    SELECT langchain_metadata->>'sha1_hash' AS sha1_hash
    FROM {settings.TABLE_NAME}
    WHERE langchain_metadata->>'sha1_hash' IN :hashes;
    ''').bindparams(sa.bindparam('hashes', expanding=True))
    async with engine.begin() as conn:
        res = await conn.execute(sql, { 'hashes': hashes })
        return { row[0] for row in res.all() }


async def load_data(dir_path: str, encoding='utf-8'):
    text_files = Path(dir_path).glob('**/*.txt')
    docs = []
    for text_file in text_files:
        loader = TextLoader(str(text_file), encoding)
        current_docs = await loader.aload()
        for d in current_docs:
            d.metadata['source'] = str(text_file)
        docs.extend(current_docs)
    if not docs:
        raise ValueError(f'No documents loaded from "{dir_path}"')
    return docs


async def split_text(
        docs: list[Document], *, chunk_size: int, chunk_overlap: int
    ):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return await asyncio.to_thread(splitter.split_documents, docs)


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def hash_document(doc: Document) -> str:
    return hash_text(doc.page_content)


if __name__ == '__main__':
    asyncio.run(main())


# TODO: Add mechanism to remove a document (`source`)
