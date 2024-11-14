from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from elasticsearch import Elasticsearch, ConnectionError
from datetime import datetime
import json
import os


class ElasticsearchSpanExporter(SpanExporter):
    """
    The exporter translates the OpenTelemetry trace data collected from your services and the metrics data collected from your applications and infrastructure to Elastic’s protocol. 
    """
    TIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
    def __init__(self, elastic_url: str = os.getenv("ELASTIC_URL", "http://localhost:9200"), export_index: str = "opentelemetry"):
        self.es = self.__connect_db__(url=elastic_url)
        self.export_index = export_index

    @staticmethod
    def __connect_db__(url: str):
        try:
            return Elasticsearch([url])
        except ConnectionError as e: 
            raise f"Failed Connecting Elastic DB - {url}, ERROR: {e}" 
        
    def export(self, spans):
        """
        Parses spans data, metrics into structured scheme. 
        """
        try:
            for span in spans:
                if not "db.system" in span.attributes.keys(): # Ignores DB writing metric spans
                    doc = {
                        "trace_id": format(span.context.trace_id, '032x'),
                        "span_id": format(span.context.span_id, '016x'),
                        "name": span.name,
                        "start_time": datetime.fromtimestamp(span.start_time // 1000000000).strftime(self.TIME_FORMAT),
                        "end_time": datetime.fromtimestamp(span.end_time // 1000000000).strftime(self.TIME_FORMAT),
                        "attributes": {k: v for k, v in span.attributes.items()},
                        "events": [
                            {"name": event.name, "attributes": {k: v for k, v in event.attributes.items()}}
                            for event in span.events
                        ],
                    }
                    self.es.index(index=self.export_index, body=json.dumps(doc))
            return SpanExportResult.SUCCESS
        except Exception as e:
            print(f"Error exporting spans to Elasticsearch: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self):
        pass  # No cleanup needed
