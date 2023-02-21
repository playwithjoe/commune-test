import logging
import os
from pprint import pprint
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http, print_answers, convert_files_to_docs
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

document_store = InMemoryDocumentStore(use_bm25=True)

doc_dir = "data/dungeons_of_drakkenheim"

all_docs = convert_files_to_docs(dir_path=doc_dir)

document_store.write_documents(all_docs)

retriever = BM25Retriever(document_store=document_store)

reader = FARMReader(model_name_or_path="deepset/minilm-uncased-squad2", use_gpu=True)

pipe = ExtractiveQAPipeline(reader, retriever)

prediction = pipe.run(
    query="Who is Monty Martin?",
    params={
        "Retriever": {"top_k": 10},
        "Reader": {"top_k": 5}
    }
)

pprint(prediction)

print_answers(
    prediction,
    details="medium" ## Choose from `minimum`, `medium`, and `all`
)
