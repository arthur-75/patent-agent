import datetime
import pickle
import typing
#ssh -i "patent_key.pem" ubuntu@ec2-18-220-33-65.us-east-2.compute.amazonaws.com
#source ~/patent-v3/venv/bin/activate
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse
import ollama
from fastapi.staticfiles import StaticFiles
# uvicorn api:app --host 0.0.0.0 --port 8080  {text.replace(/<think>[\s\S]*?<\/think>/g, "") }
#https://arxiv.org/pdf/2202.04850v1
#uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
#or gunicorn -w 2 -k uvicorn.workers.UvicornWorker api:app -b 0.0.0.0:8000
#ngrok http 8000
#    scp -i "patent_key.pem" -r "patent-v3" "ubuntu@ec2-3-138-201-99.us-east-2.compute.amazonaws.com"
#scp -i "patent_key.pem" patent-v3/docs/index.html  ubuntu@ec2-18-189-31-180.us-east-2.compute.amazonaws.com:~/patent-v3/docs/index.html

app = FastAPI(
    description="Knowledge graph.",
    title="FactGPT",
    version="0.0.1",
)

origins = [
    "http://localhost:8080",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Knowledge:
    """This class is a wrapper around the pipeline."""

    def __init__(self) -> None:
        self.pipeline = None

    def start(self):
        """Load the pipeline."""
        import os

        # Step 1: Change to the subfolder
        #os.chdir("knowledge")  # replace "folder" with your subfolder name
        print("Current directory (inside folder):", os.getcwd())

         #"database/pipeline.pkl" data/pipeline.pkl
        with open("data/pipeline.pkl", "rb") as f:
            self.pipeline = pickle.load(f)
        # Step 2: Go back to the parent directory
        #os.chdir("..")
        print("Current directory (after cd ..):", os.getcwd())
        return self

    def search(
        self,
        q: str,
        tags: str,
    ) -> typing.Dict:
        """Returns the documents."""
        return self.pipeline.search(q=q, tags=tags)

    def plot(
        self,
        q: str,
        k_tags: int,
        k_yens: int = 1,
        k_walk: int = 3,
    ) -> typing.Dict:
        """Returns the graph."""
        nodes, links = self.pipeline.plot(
            q=q,
            k_tags=k_tags,
            k_yens=k_yens,
            k_walk=k_walk,
        )
        return {"nodes": nodes, "links": links}


knowledge = Knowledge()



async def async_chat(query: str, content: str):
    """Re-rank the documents using a local Ollama model."""

    full_prompt = f"""
    You are a senior patent-search analyst.

    <task>
    - Restate the claim in one sentence.
    - For each paper, list its title, classify as:
    • Matched – fully covers all claim elements  
    • Similar   – covers most elements or if any doubt  
    • Irrelevant – no meaningful overlap  
    then give a 1–2 sentence rationale.
    </task>

    <rules>
    • Rely only on the provided text.  
    • If in doubt, classify as Similar.  
    • Do not invent content.
    </rules>

    <claim>
    {query}
    </claim>

    <papers>
    {content}
    </papers>
    """

    response = ollama.chat(
        model="qwen3:1.7b",  # or "llama3", "mistral", etc.
        messages=[{"role": "user", "content": full_prompt}],
        stream=True, options={'temperature': .5},
    )

    answer = "\n"
    for chunk in response:
        token = chunk["message"]["content"]
        answer += token
        #answer = answer
        yield answer.strip()


@app.get("/search/{sort}/{tags}/{k_tags}/{q}")
def search(k_tags: int, tags: str, sort: bool, q: str):
    """Search for documents."""
    tags = tags != "null"#tags = tags.lower() != "false" and tags.lower() != "null"#
    documents = knowledge.search(q=q, tags=tags)
    if bool(sort):
        documents = [
            document
            for _, document in sorted(
                [(document["date"], document) for document in documents],
                key=lambda document: datetime.datetime.strptime(
                    document[0], "%Y-%m-%d"
                ),
                reverse=True,
            )
        ]
    return {"documents": documents}


@app.get("/plot/{k_tags}/{q}", response_class=ORJSONResponse)
def plot(k_tags: int, q: str):
    """Plot tags."""
    return knowledge.plot(q=q, k_tags=k_tags)


@app.on_event("startup")
def start():
    """Intialiaze the pipeline."""
    return knowledge.start()


@app.get("/chat/{k_tags}/{q}")
async def chat(k_tags: int, q: str):
    """LLM recommendation."""
    documents = knowledge.search(q=q, tags=False)
    content = ""
    count=0
    for document in documents:
        count+=1
        content += f"TITLE: {count}" + document["title"] + "\n"
        content += "summary: " + document["summary"][:30] + "\n"
        content += "targs: " + (
            ", ".join(document["tags"] + document["extra-tags"]) + "\n"
        )
        #content += "url: " + document["url"] + "\n\n"
        content +='\n\n\n'
        if count>10:break
    #content = "title: ".join(content[:3000].split("title:")[:-1])
    return StreamingResponse(
        async_chat(query=q, content=content), media_type="text/plain"
    )


app.mount("/", StaticFiles(directory="docs", html=True), name="static")

