from pydantic import BaseModel

class Rotation(BaseModel):
    time: str
    retention: str

class LogFormat(BaseModel):
    simple: str
    serialize: bool

class LoggerConfig(BaseModel):
    singleton: bool
    format: str
    formatters: LogFormat
    level: str
    filename: str
    rotation: Rotation
    
    # def model_post_init(self):
    #     return self.formatters[self.format] if self.format in self.formatters.keys() else None
