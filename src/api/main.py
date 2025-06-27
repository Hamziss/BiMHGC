from fastapi import FastAPI
from .routers import extract_router,dynamic_ppi_router, protein_embeddings_router, go_enrichment_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]  # Allow all origins for CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract_router)
app.include_router(dynamic_ppi_router)
app.include_router(protein_embeddings_router)
app.include_router(go_enrichment_router)


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}