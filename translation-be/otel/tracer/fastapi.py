from otel.span.exporter.elastic import ElasticsearchSpanExporter
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor 
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


def set_otel_fastapi_tracer(app, service_name, exporter):
    """
    Define OpenTelemetry tracer on fastapi application, collects `FastAPI` app telemetry data.
    """
    # Create a TracerProvider for FastAPI
    resource = Resource(attributes={SERVICE_NAME: service_name})
    tracer_provider = TracerProvider(resource=resource)

    # Create a BatchSpanProcessor and add the exporter to it
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)

    # Set the TracerProvider as the global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Instrument FastAPI with OpenTelemetry
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
