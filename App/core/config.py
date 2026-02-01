# App/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    gcp_project_id: str = Field(..., env="GCP_PROJECT_ID")
    gcp_location: str = Field("us", env="GCP_LOCATION")
    gcp_processor_id: str = Field(..., env="GCP_PROCESSOR_ID")
    gcp_key_path: str = Field("client-docai.json", env="GCP_KEY_PATH")

    GROQ_URL:str = Field(..., env="GROQ_URL")
    GROQ_MODEL:str = Field(..., env="GROQ_MODEL")
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")

    @property
    def processor_name(self) -> str:
        """Full Document AI processor path"""
        return f"projects/{self.gcp_project_id}/locations/{self.gcp_location}/processors/{self.gcp_processor_id}"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
