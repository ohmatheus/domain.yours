from pathlib import Path
from pydantic import BaseSettings, SecretStr
from dotenv import load_dotenv

ROOT_PATH = Path(__file__).parent.parent

load_dotenv()


class DomainSettings(BaseSettings):
    HUGGINGFACE_API_TOKEN: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None
    DEVICE: str = "cpu"

    @property
    def huggingface_credentials(self) -> str:
        if self.HUGGINGFACE_API_TOKEN is None:
            raise ValueError("HUGGINGFACE_API_TOKEN is not set")
        return self.HUGGINGFACE_API_TOKEN.get_secret_value()

    @property
    def openai_credentials(self) -> str:
        if self.OPENAI_API_KEY is None:
            raise ValueError("OPENAI_API_KEY is not set")
        return self.OPENAI_API_KEY.get_secret_value()


config = DomainSettings()
