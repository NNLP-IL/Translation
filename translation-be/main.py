from fastapi import FastAPI, APIRouter
from app.routers import detect, entity_extract, transliterate, translator
from otel.span.exporter.elastic import ElasticsearchSpanExporter
from otel.tracer.fastapi import set_otel_fastapi_tracer
import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn 

SERVICE_NAME: str = os.getenv("SERVICE_NAME", "Translation")
ES_OTEL_INDEX: str = os.getenv("ELASTIC_OTEL_INDEX", "translation_opentelemetry")

# Create an instance of ElasticsearchSpanExporter
elastic_exporter = ElasticsearchSpanExporter(export_index=ES_OTEL_INDEX)

# Initialize FastAPI application
app = FastAPI(title=SERVICE_NAME)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)
# Define OpenTelemetry tracer on fastapi application
set_otel_fastapi_tracer(app=app, service_name=SERVICE_NAME, exporter=elastic_exporter)

# Define routes
prefix_router = APIRouter(prefix='/api')
prefix_router.include_router(router=detect.router, prefix='/detection')
prefix_router.include_router(router=translator.router)
prefix_router.include_router(router=transliterate.router)
prefix_router.include_router(router=entity_extract.router)
app.include_router(router=prefix_router)


@app.get(path="/", tags=["Metadata"])
async def root():
    return {
        "message": "Hello World"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, env_file="app/.env")
